import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/contextual_cpu.py"


def test_safe_import():
    spec = importlib.util.spec_from_file_location("context_optional", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_help_without_ml():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True
    )
    assert r.returncode == 0
    assert "extract" in r.stdout


def test_refuses_existing_output(tmp_path):
    path = tmp_path / "existing.npz"
    path.write_text("unchanged")
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "extract", "--output", str(path)],
        text=True,
        capture_output=True,
    )
    assert r.returncode != 0
    assert path.read_text() == "unchanged"
