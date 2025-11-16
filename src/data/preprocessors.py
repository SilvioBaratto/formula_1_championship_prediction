"""
Data preprocessing and cleaning functions for F1 race data.
"""

import pandas as pd
import numpy as np

from src.utils.config import DNF_POSITION, POINTS_SYSTEM
from src.utils.logging import get_logger

logger = get_logger(__name__)


def detect_dnf(status: str) -> bool:
    if pd.isna(status):
        return False

    status_lower = str(status).strip().lower()

    if status_lower == "finished" or status_lower == "lapped":
        return False

    if status_lower.startswith("+") and "lap" in status_lower:
        return False

    return True


def clean_race_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dnf"] = df["status"].apply(detect_dnf)
    df["finish_position_clean"] = df.apply(
        lambda row: DNF_POSITION if row["dnf"] else row["finish_position"], axis=1
    )

    df = convert_timedelta_to_seconds(df)

    df["grid_position"] = df["grid_position"].replace(0, 20)
    df["grid_position"] = df["grid_position"].fillna(20)
    df["points"] = df["points"].clip(lower=0)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["year", "round", "finish_position"])

    logger.info(
        f"Cleaned {len(df)} records, DNFs: {df['dnf'].sum()} ({df['dnf'].mean()*100:.1f}%)"
    )

    return df


def convert_timedelta_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    timedelta_columns = ["race_time"]

    for col in timedelta_columns:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col], errors="coerce")
            df[f"{col}_seconds"] = df[col].apply(
                lambda x: x.total_seconds() if pd.notna(x) else np.nan
            )

    return df


def add_qualifying_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gap_to_pole"] = df["grid_position"] - 1
    df["qualifying_percentile"] = df.groupby(["year", "round"])[
        "grid_position"
    ].transform(lambda x: x.rank(pct=True))

    return df


def calculate_race_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time_to_winner"] = df.groupby(["year", "round"])["race_time_seconds"].transform(
        lambda x: x - x.min() if x.notna().any() else np.nan
    )
    df.loc[df["dnf"], "time_to_winner"] = np.nan

    return df


def add_percentile_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["finish_percentile"] = df.groupby(["year", "round"])[
        "finish_position_clean"
    ].transform(lambda x: x.rank(pct=True))

    return df


def validate_points_consistency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def get_expected_points(position: float, dnf: bool) -> float:
        if dnf or pd.isna(position):
            return 0.0
        return POINTS_SYSTEM.get(int(position), 0.0)

    df["expected_points"] = df.apply(
        lambda row: get_expected_points(row["finish_position"], row["dnf"]), axis=1
    )

    mismatches = (df["points"] - df["expected_points"]).abs() > 0.1

    if mismatches.any():
        logger.warning(
            f"Found {mismatches.sum()} point mismatches (fastest lap bonus or penalties)"
        )

    return df


def add_dnf_rate_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["driver", "year", "round"])

    df["dnf_rate_last_n_races"] = (
        df.groupby("driver")["dnf"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    weather_cols = ["air_temp", "track_temp", "humidity", "pressure", "wind_speed"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].fillna(False)

    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting preprocessing pipeline")

    initial_rows = len(df)

    df = clean_race_data(df)
    df = add_qualifying_features(df)
    df = calculate_race_time_features(df)
    df = add_percentile_rankings(df)
    df = validate_points_consistency(df)
    df = add_dnf_rate_features(df, window=10)
    df = handle_missing_data(df)

    logger.info(
        f"Preprocessing complete: {initial_rows} → {len(df)} rows, {len(df.columns)} columns"
    )

    return df


def merge_race_and_qualifying_data(
    race_df: pd.DataFrame, qualifying_df: pd.DataFrame
) -> pd.DataFrame:
    if qualifying_df.empty:
        logger.warning("No qualifying data to merge")
        return race_df

    logger.info(f"Race data already contains grid positions: {len(race_df)} rows")
    return race_df


def save_preprocessed_data(df: pd.DataFrame, filepath):
    df.to_csv(str(filepath), index=False)
    logger.info(f"Saved preprocessed data to {filepath}")


if __name__ == "__main__":
    from src.utils.config import (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RAW_RACE_RESULTS_FILE,
        PROCESSED_DATA_FILE,
    )

    logger.info("Testing preprocessor")

    raw_data_path = RAW_DATA_DIR / RAW_RACE_RESULTS_FILE

    if raw_data_path.exists():
        df = pd.read_csv(raw_data_path)
        df_processed = preprocess_pipeline(df)

        processed_path = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE
        save_preprocessed_data(df_processed, processed_path)

        logger.info("\nSample:")
        logger.info(df_processed.head().to_string())
    else:
        logger.error(f"Raw data not found: {raw_data_path}")
