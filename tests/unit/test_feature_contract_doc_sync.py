import re
from pathlib import Path

from src.common.feature_contract import FEATURES_V1


def _doc_feature_list() -> list[str]:
    text = Path("docs/feature_contract_v1.md").read_text(encoding="utf-8")
    matches = re.findall(r"^\d+\.\s+`([^`]+)`\s*$", text, flags=re.MULTILINE)
    return matches


def test_feature_contract_doc_matches_code_order():
    assert _doc_feature_list() == FEATURES_V1
