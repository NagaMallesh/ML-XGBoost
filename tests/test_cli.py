from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import main_argparse


def test_build_parser_train_defaults() -> None:
    parser = main_argparse.build_parser()
    args = parser.parse_args(["train"])

    assert args.command == "train"
    assert args.test_size == 0.2
    assert args.random_state == 42
    assert args.tune_hyperparameters is False
    assert args.tuning_iterations == 15
    assert args.no_threshold_tuning is False
    assert args.min_precision is None
    assert args.no_smote is False
    assert args.no_feature_engineering is False
    assert args.data_path.name == "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def test_main_dispatches_download(monkeypatch) -> None:
    called = {"download": False}

    def _download_stub(data_dir):
        called["download"] = True
        return True

    monkeypatch.setattr(main_argparse, "download_kaggle_dataset", _download_stub)
    monkeypatch.setattr(sys, "argv", ["main_argparse.py", "download"])

    main_argparse.main()

    assert called["download"] is True


def test_main_dispatches_train(monkeypatch, tmp_path: Path) -> None:
    received = {}

    def _train_stub(
        data_path,
        models_dir,
        output_dir,
        test_size,
        random_state,
        tune_hyperparameters,
        tuning_iterations,
        tune_threshold,
        min_precision,
        use_smote,
        apply_feature_engineering,
    ):
        received["data_path"] = data_path
        received["models_dir"] = models_dir
        received["output_dir"] = output_dir
        received["test_size"] = test_size
        received["random_state"] = random_state
        received["tune_hyperparameters"] = tune_hyperparameters
        received["tuning_iterations"] = tuning_iterations
        received["tune_threshold"] = tune_threshold
        received["min_precision"] = min_precision
        received["use_smote"] = use_smote
        received["apply_feature_engineering"] = apply_feature_engineering

    monkeypatch.setattr(main_argparse, "run_training_pipeline", _train_stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main_argparse.py",
            "train",
            "--data-path",
            str(tmp_path / "sample.csv"),
            "--models-dir",
            str(tmp_path / "models"),
            "--output-dir",
            str(tmp_path / "output"),
            "--test-size",
            "0.3",
            "--random-state",
            "99",
            "--tune-hyperparameters",
            "--tuning-iterations",
            "8",
            "--no-threshold-tuning",
            "--min-precision",
            "0.6",
            "--no-smote",
            "--no-feature-engineering",
        ],
    )

    main_argparse.main()

    assert received["test_size"] == 0.3
    assert received["random_state"] == 99
    assert received["tune_hyperparameters"] is True
    assert received["tuning_iterations"] == 8
    assert received["tune_threshold"] is False
    assert received["min_precision"] == 0.6
    assert received["use_smote"] is False
    assert received["apply_feature_engineering"] is False
    assert received["data_path"].name == "sample.csv"


def test_build_parser_includes_test_command() -> None:
    parser = main_argparse.build_parser()
    args = parser.parse_args(["test"])

    assert args.command == "test"


def test_run_unit_tests_success(monkeypatch) -> None:
    def _run_stub(cmd, cwd, check):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(main_argparse.subprocess, "run", _run_stub)

    code = main_argparse.run_unit_tests()
    assert code == 0


def test_main_dispatches_test(monkeypatch) -> None:
    state = {"tests_called": False}

    def _tests_stub():
        state["tests_called"] = True
        return 0

    monkeypatch.setattr(main_argparse, "run_unit_tests", _tests_stub)
    monkeypatch.setattr(sys, "argv", ["main_argparse.py", "test"])

    main_argparse.main()

    assert state["tests_called"] is True
