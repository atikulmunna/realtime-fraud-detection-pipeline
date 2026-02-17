"""Run the local demo flow and print summary JSON."""

from __future__ import annotations

import json

from src.demo.demo_flow import run_demo_flow


def main() -> None:
    summary = run_demo_flow()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
