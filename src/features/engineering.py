"""
Feature engineering functions for F1 race data.
"""

import pandas as pd
import numpy as np

from typing import Optional
from src.utils.config import RECENT_RACES_WINDOW, FORM_WINDOW, DNF_WINDOW
from src.utils.logging import get_logger
from src.features.track_clustering import (
    add_track_type,
    calculate_track_type_performance,
)

logger = get_logger(__name__)


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["grid_squared"] = df["grid_position"] ** 2
    df["grid_log"] = np.log1p(df["grid_position"])

    return df


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["driver", "year", "round"])

    df["points_last_3_races"] = (
        df.groupby("driver")["points"]
        .rolling(window=RECENT_RACES_WINDOW, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["avg_finish_last_5_races"] = (
        df.groupby("driver")["finish_position"]
        .rolling(window=FORM_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["dnf_rate_last_10_races"] = (
        df.groupby("driver")["dnf"]
        .rolling(window=DNF_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def add_championship_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["driver", "year", "round"])

    df["cumulative_points"] = df.groupby(["driver", "year"])["points"].cumsum()

    df["championship_position"] = df.groupby(["year", "round"])[
        "cumulative_points"
    ].rank(method="min", ascending=False)

    return df


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["team", "year", "round"])

    df["team_avg_position_season"] = (
        df.groupby(["team", "year"])["finish_position"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    return df


def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating feature matrix...")

    df = add_track_type(df)
    logger.info("✓ Added track type")

    df = calculate_track_type_performance(df)
    logger.info("✓ Calculated track type performance")

    df = add_grid_features(df)
    logger.info("✓ Added grid features")

    df = add_recent_form_features(df)
    logger.info("✓ Added recent form features")

    df = add_championship_features(df)
    logger.info("✓ Added championship features")

    df = add_team_features(df)
    logger.info("✓ Added team features")

    logger.info(f"Feature matrix created: {len(df)} rows, {len(df.columns)} columns")

    return df


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "finish_position",
    feature_cols: Optional[list] = None,
) -> tuple:
    if feature_cols is None:
        raise ValueError("feature_cols must be provided")

    df_clean = df.dropna(subset=[target_col])

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    X = X.fillna(X.median())

    logger.info(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")

    return X, y, feature_cols


def engineer_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering pipeline...")

    df = create_feature_matrix(df)

    logger.info(f"Pipeline complete: {len(df.columns)} features available")

    return df


def engineer_prediction_features(
    prediction_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    completed_2025_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    logger.info("Engineering prediction features (hybrid mode)")

    if completed_2025_df is not None and not completed_2025_df.empty:
        logger.info(f"Using {len(completed_2025_df)} completed 2025 records")
        context_df = pd.concat([historical_df, completed_2025_df], ignore_index=True)
    else:
        logger.info("Using historical data only")
        context_df = historical_df.copy()

    pred_template = prediction_df.copy()

    if (
        "grid_position" not in pred_template.columns
        or pred_template["grid_position"].isna().all()
    ):
        driver_avg_grid = context_df.groupby("driver")["grid_position"].mean()
        pred_template["grid_position"] = (
            pred_template["driver"].map(driver_avg_grid).fillna(10.0)
        )

    pred_template["finish_position"] = np.nan
    if "points" not in pred_template.columns:
        pred_template["points"] = 0.0
    if "dnf" not in pred_template.columns:
        pred_template["dnf"] = False

    weather_cols = ["air_temp", "track_temp", "humidity", "wind_speed", "rainfall"]
    for col in weather_cols:
        if col in context_df.columns:
            if col not in pred_template.columns or pred_template[col].isna().all():
                track_weather = context_df.groupby("track_name")[col].mean()
                pred_template[col] = pred_template["track_name"].map(track_weather)
                pred_template[col] = pred_template[col].fillna(context_df[col].mean())

    combined_df = pd.concat([context_df, pred_template], ignore_index=True)

    logger.info("Running feature engineering pipeline...")
    combined_df = create_feature_matrix(combined_df)

    logger.info("Adding Bayesian driver/constructor features...")
    from src.models.bayesian import add_driver_constructor_features

    combined_df = add_driver_constructor_features(combined_df)
    logger.info("✓ Added Bayesian features")

    prediction_indices = combined_df.index[-len(pred_template) :]
    prediction_features = combined_df.loc[prediction_indices].copy()

    form_features = [
        "points_last_3_races",
        "avg_finish_last_5_races",
        "dnf_rate_last_10_races",
    ]

    for feature in form_features:
        if feature in prediction_features.columns:
            for driver in prediction_features["driver"].unique():
                driver_mask = prediction_features["driver"] == driver

                if driver in context_df["driver"].values:
                    driver_context = context_df[context_df["driver"] == driver]
                    if (
                        feature in driver_context.columns
                        and not driver_context[feature].isna().all()
                    ):
                        last_value = driver_context[feature].iloc[-1]
                        feature_values = pd.Series(
                            prediction_features.loc[driver_mask, feature]
                        )
                        prediction_features.loc[driver_mask, feature] = (
                            feature_values.fillna(last_value)
                        )

    if completed_2025_df is not None and not completed_2025_df.empty:
        current_points = completed_2025_df.groupby("driver")["points"].sum()

        for driver in prediction_features["driver"].unique():
            driver_mask = prediction_features["driver"] == driver
            if driver in current_points.index:
                prediction_features.loc[driver_mask, "cumulative_points"] = (
                    current_points[driver]
                )
            else:
                prediction_features.loc[driver_mask, "cumulative_points"] = 0.0

        for round_num in prediction_features["round"].unique():
            round_mask = prediction_features["round"] == round_num
            cumulative_pts = pd.Series(
                prediction_features.loc[round_mask, "cumulative_points"]
            )
            championship_pos = cumulative_pts.rank(method="min", ascending=False)
            prediction_features.loc[round_mask, "championship_position"] = (
                championship_pos
            )

    numeric_cols = prediction_features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if prediction_features[col].isna().any():
            fill_value = (
                context_df[col].median()
                if col in context_df.columns
                else prediction_features[col].median()
            )
            if pd.isna(fill_value):
                fill_value = 0.0
            prediction_features[col] = prediction_features[col].fillna(fill_value)

    logger.info(
        f"✓ Engineered {len(prediction_features.columns)} features for {len(prediction_features)} predictions"
    )

    return prediction_features
