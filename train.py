from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib  # type: ignore
import numpy as np

from rul.config import Config, load_config
from rul.data import get_features_and_target, read_csv, split_train_test, preprocess_data
from rul.metrics import regression_report
from rul.model import build_pipeline
from rul.utils import ensure_dir, get_logger, save_json, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Battery RUL regression model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config (optional)")
    parser.add_argument("--data", type=str, default=None, help="Override path to input CSV data")
    parser.add_argument("--models_dir", type=str, default=None, help="Directory to save trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("rul.train")

    cfg: Config = load_config(args.config)
    if args.data:
        cfg.data_path = args.data
    if args.models_dir:
        cfg.models_dir = args.models_dir

    logger.info(f"Loading data from {cfg.data_path}")
    df = read_csv(cfg.data_path)
    
    logger.info("Preprocessing data (Feature Engineering)...")
    df = preprocess_data(df)
    
    # Exclude Cycle_Index as per v2 notebook to avoid rote learning
    exclude_cols = ["Cycle_Index"]
    X, y, feature_cols = get_features_and_target(df, cfg.target_col, exclude_cols=exclude_cols)

    logger.info(f"Detected {len(feature_cols)} features: {feature_cols}")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=cfg.test_size, random_state=cfg.random_state)

    logger.info("Building pipeline and training model…")
    pipe = build_pipeline(feature_cols, cfg.model)
    pipe.fit(X_train, y_train)

    logger.info("Evaluating on test set…")
    y_pred = pipe.predict(X_test)
    report = regression_report(y_test, y_pred)
    logger.info(f"Test metrics: {json.dumps(report, indent=2)}")

    # Save artifacts
    models_dir = ensure_dir(cfg.models_dir)
    ts = timestamp()
    model_path = models_dir / f"rul_model_{ts}.joblib"
    meta_path = models_dir / f"rul_model_{ts}.meta.json"
    joblib.dump(pipe, model_path)
    save_json(
        {
            "config": cfg.to_dict(),
            "features": feature_cols,
            "metrics": report,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        meta_path,
    )
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
