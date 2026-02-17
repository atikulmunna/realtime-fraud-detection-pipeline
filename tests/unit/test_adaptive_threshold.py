from src.streaming.adaptive_threshold import compute_adaptive_threshold


def test_compute_adaptive_threshold_returns_current_when_insufficient_samples():
    scores = [0.1, 0.2, 0.3]
    t = compute_adaptive_threshold(scores, current_threshold=0.55, min_samples=10)
    assert t == 0.55


def test_compute_adaptive_threshold_quantile_with_bounds():
    scores = [i / 100 for i in range(100)]
    t = compute_adaptive_threshold(scores, quantile=0.9, min_samples=10, floor=0.1, ceiling=0.8)
    assert 0.1 <= t <= 0.8
    assert t == 0.8
