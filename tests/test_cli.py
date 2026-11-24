import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from ner_bilstm.cli import app

runner = CliRunner()


def make_data(path: Path) -> None:
    path.write_text("Sentence #,Word,Tag\ns1,a,O\n,b,O\ns2,c,O\ns3,d,O\ns4,e,O\n")


def test_help_import_has_no_tensorflow() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ner_bilstm.cli; assert 'tensorflow' not in sys.modules",
        ],
        capture_output=True,
    )
    assert result.returncode == 0
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["train", "--help"]).exit_code == 0


def test_check_config_and_precedence(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    make_data(data)
    config = tmp_path / ".env"
    config.write_text(f"NER_DATA_PATH={data}\n")
    result = runner.invoke(app, ["check-data", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "sentences": 4,
        "train_sentences": 3,
        "validation_sentences": 1,
        "training_words": 4,
        "training_tags": 1,
        "validation_oov_tokens": 1,
    }
    config.write_text("NER_DATA_PATH=/missing.csv\n")
    assert (
        runner.invoke(
            app, ["check-data", "--data", str(data), "--config", str(config)]
        ).exit_code
        == 0
    )


def test_missing_config_and_data(tmp_path: Path) -> None:
    for args in [
        [],
        ["--config", str(tmp_path / "missing")],
        ["--data", str(tmp_path / "missing")],
    ]:
        result = runner.invoke(app, ["check-data", *args])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
    empty = tmp_path / ".env"
    empty.write_text("OTHER=value\n")
    assert runner.invoke(app, ["check-data", "--config", str(empty)]).exit_code == 2


def test_training_does_not_overwrite(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    make_data(data)
    result = runner.invoke(app, ["train", str(data), str(tmp_path)])
    assert result.exit_code == 2
    result = runner.invoke(app, ["predict", str(tmp_path), "hello"])
    assert result.exit_code == 2
