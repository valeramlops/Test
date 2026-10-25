"""Conservative post-processing for Russian passport OCR.

Usage:
    python postprocess.py data/test_predictions.jsonl answers.jsonl
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys

FIELDS = (
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
    "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
    "Issue_organization_ru",
)
DATE_FIELDS = ("Birth_date", "Issue_date")
NAME_FIELDS = ("Last_name_ru", "First_name_ru", "Middle_name_ru")
DIGIT_WEIGHTS = (7, 3, 1)
CYRILLIC_TO_MRZ = str.maketrans({
    # Russian internal-passport MRZ transliteration (not the international
    # passport spelling): several Cyrillic letters are represented by digits.
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "2",
    "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3", "Ш": "4", "Щ": "5",
    "Ъ": "6", "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7", "Я": "8",
})
HIGH_RISK_MRZ_NAME_GLYPHS = frozenset("ЖЙЯ")


def clean(value: str) -> str:
    return " ".join(value.split()).upper()


def compact(value: str) -> str:
    return re.sub(r"[^0-9A-ZА-Я]", "", clean(value))


def value_and_conf(field: dict | None) -> tuple[str, float]:
    if not field or not field.get("lines"):
        return "", 0.0
    lines = field["lines"]
    text = clean(" ".join(str(line.get("text") or "") for line in lines))
    chars = [score for line in lines for score in (line.get("char_conf") or [])]
    # The weakest character is what matters for accepting a fully correct document.
    confidence = min(chars) if chars else float(field.get("conf") or 0)
    return text, confidence


def checksum(value: str) -> int:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
    return sum(alphabet.index(char) * DIGIT_WEIGHTS[index % 3]
               for index, char in enumerate(value)) % 10


def parse_date(value: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(value, "%d.%m.%Y").date()
    except ValueError:
        return None


def valid_visual_formats(fields: dict[str, str]) -> bool:
    if fields["Sex_ru"] not in {"МУЖ.", "ЖЕН."}:
        return False
    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields["Birth_date"]):
        return False
    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields["Issue_date"]):
        return False
    if not re.fullmatch(r"\d{3}-\d{3}", fields["Issue_organisation_code"]):
        return False
    # DaData is useful as a reference, but not a complete legal registry:
    # a missing historical code must not turn a readable valid passport into a reject.
    if not re.fullmatch(r"\d{2} \d{2} \d{6}", fields["Licence_number"]):
        return False
    return all(fields[name] for name in NAME_FIELDS + ("Birth_place_ru", "Issue_organization_ru"))


def anniversary(date: dt.date, years: int) -> dt.date:
    """Return a calendar anniversary, including the 29 February edge case."""
    try:
        return date.replace(year=date.year + years)
    except ValueError:
        return date.replace(year=date.year + years, month=2, day=28)


def valid_dates(fields: dict[str, str]) -> bool:
    birth, issued = (parse_date(fields[name]) for name in DATE_FIELDS)
    as_of = dt.date(2026, 10, 1)
    if birth is None or issued is None or not birth <= issued <= as_of:
        return False
    if issued < anniversary(birth, 14) or issued > anniversary(birth, 100):
        return False

    # A Russian internal passport must be replaced within 90 days after 20 and 45.
    # Its issue date tells which replacement cycle the document belongs to.
    twentieth, forty_fifth = anniversary(birth, 20), anniversary(birth, 45)
    if issued < twentieth:
        expires = twentieth + dt.timedelta(days=90)
    elif issued < forty_fifth:
        expires = forty_fifth + dt.timedelta(days=90)
    else:
        expires = None
    return expires is None or as_of <= expires


def mrz_checks(mrz: dict | None, fields: dict[str, str]) -> bool:
    """Use only unambiguous ICAO checks; missing MRZ does not itself reject."""
    if not mrz or len(mrz.get("lines") or []) != 2:
        return True

    first, second = (str(line.get("text") or "").replace(" ", "") for line in mrz["lines"])
    if len(first) != 44 or len(second) != 44:
        return False
    if not first.startswith("PNRUS"):
        return False

    # Document number and birth date have ISO 7501 check digits in this layout.
    if not (second[9].isdigit() and second[19].isdigit()):
        return False
    if checksum(second[:9]) != int(second[9]):
        return False
    if checksum(second[13:19]) != int(second[19]):
        return False

    birth = parse_date(fields["Birth_date"])
    issued = parse_date(fields["Issue_date"])
    if birth is None or issued is None:
        return False
    if second[13:19] != birth.strftime("%y%m%d"):
        return False
    if second[29:35] != issued.strftime("%y%m%d"):
        return False

    # MRZ stores three series digits and all six number digits: the fourth
    # visual series digit is intentionally omitted by the Russian layout.
    licence = compact(fields["Licence_number"])
    expected_number = licence[:3] + licence[4:]
    if second[:9] != expected_number:
        return False
    if second[35:41] != compact(fields["Issue_organisation_code"]):
        return False

    sex = "M" if fields["Sex_ru"] == "МУЖ." else "F"
    return second[20] == sex


def mrz_names_agree(mrz: dict | None, fields: dict[str, str]) -> bool:
    """Cross-check names when the MRZ is structurally readable.

    This is deliberately approximate because Russian-to-MRZ transliteration has
    approved variants. It still catches a disagreement rather than repairing it.
    """
    if not mrz or len(mrz.get("lines") or []) != 2:
        return True
    first = str(mrz["lines"][0].get("text") or "").replace(" ", "")
    if len(first) != 44 or not first.startswith("PNRUS"):
        return False
    visual = [clean(fields[name]).translate(CYRILLIC_TO_MRZ) for name in NAME_FIELDS]
    mrz_parts = [part for part in first[5:].split("<") if part]
    # The surname is the first part, then a single MRZ part can contain all names.
    joined = "".join(mrz_parts)
    return all(name in joined for name in visual if name)


def has_structural_mrz(prediction: dict) -> bool:
    mrz = prediction.get("mrz") or {}
    lines = mrz.get("lines") or []
    if len(lines) != 2:
        return False
    first = str(lines[0].get("text") or "").replace(" ", "")
    second = str(lines[1].get("text") or "").replace(" ", "")
    return len(first) == len(second) == 44 and first.startswith("PNRUS")


def birth_place_is_complete(prediction: dict) -> bool:
    """Reject an apparent OCR truncation in a visibly multi-line field."""
    field = (prediction.get("viz") or {}).get("Birth_place_ru") or {}
    lines = field.get("lines") or []
    box = field.get("box") or []
    if len(lines) != 1 or len(box) != 4 or box[3] < 60:
        return True
    line_box = lines[0].get("box") or []
    # There is room for a second line, while the sole read ends well before
    # the right edge: this is a common signature of an omitted locality suffix.
    return len(line_box) != 4 or line_box[2] >= box[2] * 0.90


def mrz_name_glyphs_are_safe(prediction: dict, fields: dict[str, str]) -> bool:
    """Reject correlated visual/MRZ OCR errors for unstable name glyphs."""
    if not has_structural_mrz(prediction):
        return True
    names = "".join(fields[name] for name in NAME_FIELDS)
    return not (set(names) & HIGH_RISK_MRZ_NAME_GLYPHS)


def confidently_read(prediction: dict, fields: dict[str, str], confidences: dict[str, float]) -> bool:
    # The threshold is applied to the weakest character, not to a field average.
    # A structurally readable MRZ independently verifies six visual requisites;
    # without it, retain only near-perfect visual reads. Fields not carried by
    # MRZ keep their own stricter thresholds.
    threshold = 0.950 if has_structural_mrz(prediction) else 0.999
    return (
        birth_place_is_complete(prediction)
        and mrz_name_glyphs_are_safe(prediction, fields)
        and confidences["Licence_number"] >= 0.992
        and confidences["Birth_place_ru"] >= 0.992
        and confidences["Issue_organization_ru"] >= 0.975
        and all(confidences[name] >= threshold for name in FIELDS)
    )


def process(prediction: dict) -> dict:
    values_and_confidences = {
        name: value_and_conf((prediction.get("viz") or {}).get(name)) for name in FIELDS
    }
    fields = {name: item[0] for name, item in values_and_confidences.items()}
    confidences = {name: item[1] for name, item in values_and_confidences.items()}

    accepted = (
        confidently_read(prediction, fields, confidences)
        and valid_visual_formats(fields)
        and valid_dates(fields)
        and mrz_checks(prediction.get("mrz"), fields)
        and mrz_names_agree(prediction.get("mrz"), fields)
    )
    result = {"doc_id": prediction["doc_id"], "decision": "accept" if accepted else "reject"}
    if accepted:
        result["fields"] = fields
    return result


def main(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(json.dumps(process(json.loads(line)), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python postprocess.py <predictions.jsonl> <answers.jsonl>")
    main(*sys.argv[1:])
