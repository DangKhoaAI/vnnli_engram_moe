from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from vnnli_engram_moe.constants import LABELS


def compute_classification_metrics(predictions: Iterable[int], labels: Iterable[int]) -> dict[str, float]:
    predicted = np.array(list(predictions))
    gold = np.array(list(labels))
    if predicted.shape != gold.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    accuracy = float((predicted == gold).mean()) if len(gold) else 0.0
    class_f1_scores: list[float] = []
    metrics: dict[str, float] = {"accuracy": accuracy}

    for label_id, label_name in enumerate(LABELS):
        true_positive = int(((predicted == label_id) & (gold == label_id)).sum())
        false_positive = int(((predicted == label_id) & (gold != label_id)).sum())
        false_negative = int(((predicted != label_id) & (gold == label_id)).sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        metrics[f"{label_name}_precision"] = float(precision)
        metrics[f"{label_name}_recall"] = float(recall)
        metrics[f"{label_name}_f1"] = float(f1)
        class_f1_scores.append(f1)

    metrics["macro_f1"] = float(sum(class_f1_scores) / len(class_f1_scores))
    metrics["support"] = float(len(gold))
    prediction_distribution = Counter(predicted.tolist())
    for label_id, label_name in enumerate(LABELS):
        metrics[f"predicted_{label_name}"] = float(prediction_distribution.get(label_id, 0))
    return metrics

