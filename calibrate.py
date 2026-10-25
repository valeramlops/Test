"""Explore conservative confidence policies on the labelled train set.

This script is diagnostic only; postprocess.py remains the submission entry point.
"""
import json

import postprocess as p

# (generic MRZ-backed threshold, passport number, birth place, issuer name)
# The last three fields have components not independently carried in MRZ.
POLICIES = (
    (0.975, 0.992, 0.992, 0.000),
    (0.970, 0.992, 0.992, 0.975),
    (0.965, 0.992, 0.992, 0.975),
    (0.960, 0.992, 0.992, 0.975),
    (0.950, 0.992, 0.992, 0.975),
)


def extract(prediction: dict) -> tuple[dict[str, str], dict[str, float]]:
    pairs = {name: p.value_and_conf((prediction.get("viz") or {}).get(name)) for name in p.FIELDS}
    return ({name: value for name, (value, _) in pairs.items()},
            {name: confidence for name, (_, confidence) in pairs.items()})


def valid(prediction: dict, fields: dict[str, str]) -> bool:
    return (
        p.valid_visual_formats(fields)
        and p.valid_dates(fields)
        and p.mrz_checks(prediction.get("mrz"), fields)
        and p.mrz_names_agree(prediction.get("mrz"), fields)
        and p.mrz_name_glyphs_are_safe(prediction, fields)
        and p.birth_place_is_complete(prediction)
    )


def accepted(row: dict, policy: tuple[float, float, float, float]) -> bool:
    generic, licence, birth_place, issuer = policy
    generic = generic if row["has_mrz"] else 0.999
    confidences = row["confidences"]
    return (
        row["valid"]
        and all(value >= generic for value in confidences.values())
        and confidences["Licence_number"] >= licence
        and confidences["Birth_place_ru"] >= birth_place
        and confidences["Issue_organization_ru"] >= issuer
    )


def wrong_fields(row: dict) -> list[str]:
    return [name for name in p.FIELDS
            if p.clean(row["fields"][name]) != p.clean(row["label"][name])]


def main() -> None:
    labels = {row["doc_id"]: row["fields"]
              for row in map(json.loads, open("data/train_labels.jsonl", encoding="utf-8"))}
    rows = []
    for prediction in map(json.loads, open("data/train_predictions.jsonl", encoding="utf-8")):
        fields, confidences = extract(prediction)
        rows.append({
            "doc_id": prediction["doc_id"],
            "fold": int(prediction["doc_id"].rsplit("_", 1)[1]) % 2,
            "has_mrz": p.has_structural_mrz(prediction),
            "fields": fields, "confidences": confidences,
            "label": labels[prediction["doc_id"]],
            "valid": valid(prediction, fields),
        })

    print("generic licence place  issuer  fold-0 good/bad  fold-1 good/bad  total")
    for policy in POLICIES:
        by_fold = []
        for fold in (0, 1):
            selected = [row for row in rows if row["fold"] == fold and accepted(row, policy)]
            good = sum(not wrong_fields(row) for row in selected)
            by_fold.append((good, len(selected) - good))
        total = sum(good + bad for good, bad in by_fold)
        print(f"{policy[0]:7.3f} {policy[1]:.3f}  {policy[2]:.3f}  {policy[3]:.3f}  "
              f"{by_fold[0][0]:3}/{by_fold[0][1]:1}           "
              f"{by_fold[1][0]:3}/{by_fold[1][1]:1}       {total:3}")

    print("\nErrors in policies with false accepts:")
    for policy in POLICIES:
        errors = [row for row in rows if accepted(row, policy) and wrong_fields(row)]
        if not errors:
            continue
        print(f"policy={policy}")
        for row in errors:
            details = ", ".join(f"{name}={row['confidences'][name]:.4f}"
                                for name in wrong_fields(row))
            print(f"  {row['doc_id']} (fold {row['fold']}): {details}")


if __name__ == "__main__":
    main()
