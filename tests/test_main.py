from __future__ import annotations

from pathlib import Path
import builtins
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import main


def test_run_unit_tests_success(monkeypatch) -> None:
    def _run_stub(cmd, cwd, check):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(main.subprocess, "run", _run_stub)

    code = main.run_unit_tests()

    assert code == 0


def test_main_dispatches_download_and_exit(monkeypatch) -> None:
    state = {"download_called": False}

    def _download_stub():
        state["download_called"] = True
        return True

    inputs = iter(["1", "4"])
    monkeypatch.setattr(main, "download_kaggle_dataset", _download_stub)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    main.main()

    assert state["download_called"] is True


def test_main_dispatches_train_and_exit(monkeypatch, tmp_path: Path) -> None:
    state = {"train_called": False}

    def _train_stub(data_path, models_dir, output_dir):
        state["train_called"] = True
        assert data_path.name == "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        assert models_dir.name == "models"
        assert output_dir.name == "output"

    inputs = iter(["2", "4"])
    monkeypatch.setattr(main, "run_training_pipeline", _train_stub)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    main.main()

    assert state["train_called"] is True


def test_main_dispatches_tests_and_exit(monkeypatch) -> None:
    state = {"tests_called": False}

    def _tests_stub():
        state["tests_called"] = True
        return 0

    inputs = iter(["3", "4"])
    monkeypatch.setattr(main, "run_unit_tests", _tests_stub)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    main.main()

    assert state["tests_called"] is True
