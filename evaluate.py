from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib  # type: ignore

from rul.config import load_config
from rul.data import get_features_and_target, read_csv, split_train_test, preprocess_data
from rul.metrics import regression_report
from rul.utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Battery RUL model on the dataset test split")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config (optional)")
    parser.add_argument("--data", type=str, default=None, help="Override path to input CSV data")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("rul.evaluate")

    cfg = load_config(args.config)
    if args.data:
        cfg.data_path = args.data

    logger.info(f"Loading model from {args.model}")
    model = joblib.load(args.model)

    logger.info(f"Loading data from {cfg.data_path}")
    df = read_csv(cfg.data_path)
    
    # Apply preprocessing to generate features
    df = preprocess_data(df)
    
    # Exclude Cycle_Index as in training
    exclude_cols = ["Cycle_Index"]
    X, y, feature_cols = get_features_and_target(df, cfg.target_col, exclude_cols=exclude_cols)
    
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=cfg.test_size, random_state=cfg.random_state)

    logger.info("Scoring on test set…")
    y_pred = model.predict(X_test)
    report = regression_report(y_test, y_pred)
    logger.info(f"Test metrics: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    main()
