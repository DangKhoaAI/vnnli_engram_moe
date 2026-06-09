import pytest

from vnnli_engram_moe.data.labels import id_to_label, label_to_id, normalize_label


def test_label_mapping_round_trip() -> None:
    assert label_to_id("entailment") == 0
    assert label_to_id("contradiction") == 1
    assert label_to_id("neutral") == 2
    assert id_to_label(0) == "entailment"
    assert normalize_label(" Neutral ") == "neutral"


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError):
        label_to_id("unsupported")

