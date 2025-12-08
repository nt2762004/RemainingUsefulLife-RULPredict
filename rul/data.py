from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {p}")
    df = pd.read_csv(p)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering and cleaning steps consistent with battery_rul_modelingv2.ipynb
    """
    df = df.copy()
    
    # Sort by Cycle_Index to ensure time-series features are correct
    if 'Cycle_Index' in df.columns:
        df = df.sort_values('Cycle_Index')

    # 1. Efficiency Ratio
    if 'Discharge Time (s)' in df.columns and 'Charging time (s)' in df.columns:
        df['Efficiency_Ratio'] = df['Discharge Time (s)'] / df['Charging time (s)']

    # 2. Voltage Drop Rate
    if 'Max. Voltage Dischar. (V)' in df.columns and 'Min. Voltage Charg. (V)' in df.columns and 'Discharge Time (s)' in df.columns:
        voltage_range = df['Max. Voltage Dischar. (V)'] - df['Min. Voltage Charg. (V)']
        df['Voltage_Drop_Rate'] = voltage_range / df['Discharge Time (s)']

    # 3. Discharge Drop Rate
    if 'Discharge Time (s)' in df.columns:
        df['Discharge_Drop_Rate'] = df['Discharge Time (s)'].diff()

    # 4. Rolling Mean (Window=10)
    if 'Discharge Time (s)' in df.columns:
        df['Discharge_Rolling_Mean_10'] = df['Discharge Time (s)'].rolling(window=10).mean()

    # 5. Rolling Std (Window=10)
    if 'Discharge Time (s)' in df.columns:
        df['Discharge_Rolling_Std_10'] = df['Discharge Time (s)'].rolling(window=10).std()

    # 6. Lag Feature (Prev_Cycle_Discharge)
    if 'Discharge Time (s)' in df.columns:
        df['Prev_Cycle_Discharge'] = df['Discharge Time (s)'].shift(1)

    # Handle NaNs created by lag/rolling features
    df = df.bfill()
    
    if 'Voltage_Drop_Rate' in df.columns:
        error_mask = df['Voltage_Drop_Rate'] < 0
        df = df[~error_mask].copy()

    return df


def get_features_and_target(
    df: pd.DataFrame, 
    target_col: str, 
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in dataset columns: {list(df.columns)}")
    
    if exclude_cols is None:
        exclude_cols = []
        
    feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
