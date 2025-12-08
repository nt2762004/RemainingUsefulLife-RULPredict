from __future__ import annotations

import argparse
from pathlib import Path

import joblib  # type: ignore
import pandas as pd

from rul.utils import ensure_dir, get_logger
from rul.data import preprocess_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained Battery RUL model on a CSV file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .joblib")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV with feature columns")
    parser.add_argument("--output_csv", type=str, default="outputs/predictions.csv", help="Where to save predictions CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("rul.predict")

    model = joblib.load(args.model)
    df = pd.read_csv(args.input_csv)
    
    logger.info("Preprocessing data (Feature Engineering)...")
    # Apply the same preprocessing as training (feature engineering + cleaning)
    df_processed = preprocess_data(df)

    # Prepare features for the model
    # Drop RUL if exists (target), and Cycle_Index (not used in v2 model)
    drop_cols = ["RUL", "Cycle_Index"]
    
    df_features = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns])

    preds = model.predict(df_features)
    out = df_processed.copy()
    out["RUL_pred"] = preds

    # Reorder columns to place RUL and RUL_pred at the end
    cols = [c for c in out.columns if c not in ["RUL", "RUL_pred"]]
    if "RUL" in out.columns:
        cols.append("RUL")
    cols.append("RUL_pred")
    out = out[cols]

    out_path = Path(args.output_csv)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
