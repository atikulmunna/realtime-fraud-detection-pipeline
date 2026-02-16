from src.common.feature_contract import FEATURES_V1, LEAKAGE_FIELDS


def test_feature_contract_has_no_leakage_fields():
    lowered = {f.lower() for f in FEATURES_V1}
    blocked = {f.lower() for f in LEAKAGE_FIELDS}
    assert lowered.isdisjoint(blocked)


def test_feature_contract_is_stable_non_empty():
    assert isinstance(FEATURES_V1, list)
    assert len(FEATURES_V1) >= 5
