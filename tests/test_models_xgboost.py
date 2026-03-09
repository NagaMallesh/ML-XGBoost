from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from models.xgboost_model import build_xgboost_model


def test_build_xgboost_model_params() -> None:
    model = build_xgboost_model(random_state=123)
    params = model.get_params()

    assert params["objective"] == "binary:logistic"
    assert params["random_state"] == 123
    assert params["eval_metric"] == "auc"
