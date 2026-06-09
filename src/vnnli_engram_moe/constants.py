"""Stable project constants."""

LABELS = ["entailment", "contradiction", "neutral"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for index, label in enumerate(LABELS)}
REQUIRED_COLUMNS = ("uid", "premise", "hypothesis", "label")

