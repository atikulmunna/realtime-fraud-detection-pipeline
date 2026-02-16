import json
from pathlib import Path


def test_event_schema_required_fields_present():
    schema_path = Path("schemas/event_v1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = set(schema.get("required", []))
    assert {
        "event_id",
        "timestamp",
        "user_id",
        "type",
        "amount",
        "old_balance_orig",
        "new_balance_orig",
    }.issubset(required)
