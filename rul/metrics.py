from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(metrics.mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(metrics.r2_score(y_true, y_pred))


def regression_report(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
