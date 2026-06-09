from __future__ import annotations

from vnnli_engram_moe.constants import ID2LABEL, LABEL2ID


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in LABEL2ID:
        raise ValueError(
            f"Unknown NLI label '{label}'. Expected one of: {', '.join(LABEL2ID)}."
        )
    return normalized


def label_to_id(label: str) -> int:
    return LABEL2ID[normalize_label(label)]


def id_to_label(label_id: int) -> str:
    if label_id not in ID2LABEL:
        raise ValueError(f"Unknown label id '{label_id}'. Expected one of: {sorted(ID2LABEL)}.")
    return ID2LABEL[label_id]

