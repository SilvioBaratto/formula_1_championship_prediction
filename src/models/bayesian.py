"""
Bayesian variance decomposition for F1 race results.
"""

import numpy as np
import pandas as pd
from typing import Dict

from src.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_variance_decomposition(
    df: pd.DataFrame,
    position_col: str = "finish_position",
    driver_col: str = "driver",
    team_col: str = "team",
) -> Dict[str, float]:
    df_clean = df[[position_col, driver_col, team_col]].dropna()

    total_variance = df_clean[position_col].var()

    team_means = df_clean.groupby(team_col)[position_col].mean()
    grand_mean = df_clean[position_col].mean()
    team_counts = df_clean.groupby(team_col).size()

    between_team_variance = ((team_means - grand_mean) ** 2 * team_counts).sum() / len(
        df_clean
    )
    within_team_variance = total_variance - between_team_variance

    constructor_contribution = (between_team_variance / total_variance) * 100
    driver_contribution = (within_team_variance / total_variance) * 100

    driver_team_performance = df_clean.groupby([driver_col, team_col])[
        position_col
    ].mean()
    team_avg_performance = df_clean.groupby(team_col)[position_col].mean()

    driver_adjustments = []
    for driver_team_key in driver_team_performance.index:
        driver, team = driver_team_key
        avg_pos = driver_team_performance[driver_team_key]
        team_avg = team_avg_performance[team]
        adjustment = team_avg - avg_pos
        driver_adjustments.append(
            {
                "driver": driver,
                "team": team,
                "avg_finish": avg_pos,
                "team_avg": team_avg,
                "skill_adjustment": adjustment,
            }
        )

    driver_skill_df = pd.DataFrame(driver_adjustments)

    results = {
        "total_variance": total_variance,
        "between_team_variance": between_team_variance,
        "within_team_variance": within_team_variance,
        "constructor_contribution_pct": constructor_contribution,
        "driver_contribution_pct": driver_contribution,
        "driver_skill_adjustments": driver_skill_df,
    }

    logger.info(
        f"Variance: Constructor {constructor_contribution:.1f}%, Driver {driver_contribution:.1f}%"
    )

    top_drivers = driver_skill_df.nlargest(10, "skill_adjustment")
    logger.info(f"\nTop drivers vs team avg:")
    for _, row in top_drivers.iterrows():
        logger.info(
            f"  {row['driver']} ({row['team']}): +{row['skill_adjustment']:.2f}"
        )

    return results


def add_driver_constructor_features(
    df: pd.DataFrame,
    position_col: str = "finish_position",
    driver_col: str = "driver",
    team_col: str = "team",
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["year", "round"])

    df["driver_avg_position_historical"] = (
        df.groupby(driver_col)[position_col]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["team_avg_position_historical"] = (
        df.groupby(team_col)[position_col]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["driver_skill_vs_team"] = (
        df["team_avg_position_historical"] - df["driver_avg_position_historical"]
    )

    df["team_strength"] = 20 - df["team_avg_position_historical"]

    return df


def analyze_driver_team_combinations(
    df: pd.DataFrame,
    year: int,
    driver_col: str = "driver",
    team_col: str = "team",
    position_col: str = "finish_position",
) -> pd.DataFrame:
    year_data = df[df["year"] == year].copy()

    if year_data.empty:
        logger.warning(f"No data for year {year}")
        return pd.DataFrame()

    combinations = (
        year_data.groupby([driver_col, team_col])
        .agg({position_col: ["mean", "std", "count"], "points": "sum"})
        .round(2)
    )

    combinations.columns = ["avg_position", "std_position", "races", "total_points"]
    combinations = combinations.reset_index()
    combinations = combinations.sort_values("avg_position")

    logger.info(
        f"\n{year} driver-team combinations:\n{combinations.to_string(index=False)}"
    )

    return combinations


def estimate_driver_value(
    df: pd.DataFrame,
    driver: str,
    driver_col: str = "driver",
    team_col: str = "team",
    position_col: str = "finish_position",
) -> Dict:
    driver_data = df[df[driver_col] == driver].copy()

    if driver_data.empty:
        return {}

    driver_avg = driver_data[position_col].mean()

    teams = driver_data[team_col].unique()
    team_baselines = []

    for team in teams:
        team_others = df[(df[team_col] == team) & (df[driver_col] != driver)]
        if not team_others.empty:
            team_baselines.append(team_others[position_col].mean())

    avg_team_baseline = np.mean(team_baselines) if team_baselines else driver_avg
    driver_value = avg_team_baseline - driver_avg

    return {
        "driver": driver,
        "avg_position": driver_avg,
        "team_baseline": avg_team_baseline,
        "driver_value": driver_value,
        "races": len(driver_data),
    }
