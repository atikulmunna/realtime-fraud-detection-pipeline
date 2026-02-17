"""Helpers for local demo execution commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoRunConfig:
    conda_env: str = "realtime-fraud"
    compose_file: str = "infra/docker-compose.yml"
    with_compose: bool = False


def build_demo_commands(config: DemoRunConfig) -> list[str]:
    cmds: list[str] = []
    if config.with_compose:
        cmds.append(f"docker compose -f {config.compose_file} up -d")
    cmds.append(f"conda run -n {config.conda_env} python scripts/demo_run.py")
    return cmds
