from __future__ import annotations

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelParams


def build_preprocessor(numeric_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_regressor(params: Optional[ModelParams] = None) -> RandomForestRegressor:
    mp = params or ModelParams()
    return RandomForestRegressor(
        n_estimators=mp.n_estimators,
        max_depth=mp.max_depth,
        min_samples_split=mp.min_samples_split,
        min_samples_leaf=mp.min_samples_leaf,
        n_jobs=mp.n_jobs,
        random_state=mp.random_state,
    )


def build_pipeline(numeric_features: List[str], params: Optional[ModelParams] = None) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features)
    reg = build_regressor(params)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", reg),
    ])
    return pipe
