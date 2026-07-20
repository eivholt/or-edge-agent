from pathlib import Path

from apps.detector.inference import ROOT_DIR, default_model_path


def test_x86_64_uses_tracked_default(monkeypatch):
    monkeypatch.delenv("EI_MODEL_PATH", raising=False)

    assert default_model_path("x86_64") == ROOT_DIR / "models" / "modelfile.eim"


def test_aarch64_uses_arm_runner(monkeypatch):
    monkeypatch.delenv("EI_MODEL_PATH", raising=False)

    assert default_model_path("aarch64") == (
        ROOT_DIR / "models" / "modelfile.aarch64.eim"
    )


def test_environment_override_takes_precedence(monkeypatch, tmp_path):
    model_path = tmp_path / "custom.eim"
    monkeypatch.setenv("EI_MODEL_PATH", str(model_path))

    assert default_model_path("aarch64") == Path(model_path).resolve()