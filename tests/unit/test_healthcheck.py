from src.observability.healthcheck import run_healthchecks


def test_run_healthchecks_success():
    def fake_fetch(url: str, timeout_s: float):
        if url.endswith("/health"):
            return 200, '{"status":"ok"}'
        if url.endswith("/metrics"):
            return 200, "feedback_requests_total 1\n"
        raise RuntimeError("unexpected url")

    res = run_healthchecks(base_url="http://x", fetch_fn=fake_fetch)
    assert res["overall_ok"] is True
    assert res["health"]["ok"] is True
    assert res["metrics"]["ok"] is True


def test_run_healthchecks_failure():
    def fake_fetch(url: str, timeout_s: float):
        if url.endswith("/health"):
            return 500, "err"
        raise RuntimeError("down")

    res = run_healthchecks(base_url="http://x", fetch_fn=fake_fetch)
    assert res["overall_ok"] is False
    assert res["health"]["ok"] is False
    assert res["metrics"]["ok"] is False
