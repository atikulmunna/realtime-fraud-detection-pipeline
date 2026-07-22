import json
from pathlib import Path

from jsonschema import Draft202012Validator


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


def test_event_and_dlq_schemas_are_valid_draft_2020_12():
    for path in (Path("schemas/event_v1.json"), Path("schemas/dlq_v1.json")):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))
