import json
from pathlib import Path

from src.common.feature_contract import FEATURES_V1


def test_sample_feedback_fixture_is_valid_jsonl():
    path = Path("data/samples/feedback.sample.jsonl")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 2

    for ln in lines:
        obj = json.loads(ln)
        assert obj["label"] in {"true_positive", "false_positive"}
        feats = obj["features"]
        for key in FEATURES_V1:
            assert key in feats
