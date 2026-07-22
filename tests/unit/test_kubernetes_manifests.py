from pathlib import Path


def test_kustomize_base_has_production_safety_resources():
    root = Path("deploy/kubernetes/base")
    kustomization = (root / "kustomization.yaml").read_text(encoding="utf-8")
    combined = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.yaml"))

    assert "flink-deployment.yaml" in kustomization
    assert "kind: FlinkDeployment" in combined
    assert "upgradeMode: savepoint" in combined
    assert "kind: HorizontalPodAutoscaler" in combined
    assert "kind: PodDisruptionBudget" in combined
    assert "kind: NetworkPolicy" in combined
    assert "readOnlyRootFilesystem: true" in combined
    assert "resources:" in combined
    assert "kafka.example.invalid" in combined
    assert "kind: Deployment\nmetadata:\n  name: postgres" not in combined
