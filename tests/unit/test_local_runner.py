from src.demo.local_runner import DemoRunConfig, build_demo_commands


def test_build_demo_commands_without_compose():
    cmds = build_demo_commands(DemoRunConfig(conda_env="realtime-fraud", with_compose=False))
    assert cmds == ["conda run -n realtime-fraud python scripts/demo_run.py"]


def test_build_demo_commands_with_compose():
    cmds = build_demo_commands(
        DemoRunConfig(conda_env="realtime-fraud", compose_file="infra/docker-compose.yml", with_compose=True)
    )
    assert cmds[0] == "docker compose -f infra/docker-compose.yml up -d"
    assert cmds[1] == "conda run -n realtime-fraud python scripts/demo_run.py"
