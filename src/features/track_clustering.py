"""
Track clustering and classification system.
"""

import pandas as pd
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

TRACK_CLUSTERS = {
    "high_speed_low_downforce": [
        "Monza",
        "Jeddah",
        "Spa-Francorchamps",
        "Baku",
        "Las Vegas",
    ],
    "street_circuit_high_downforce": [
        "Monaco",
        "Marina Bay",
        "Miami Gardens",
        "Miami",
    ],
    "high_downforce_technical": [
        "Suzuka",
        "Budapest",
        "Barcelona",
        "Zandvoort",
        "Imola",
    ],
    "medium_downforce_balanced": [
        "Silverstone",
        "Melbourne",
        "Spielberg",
        "Shanghai",
        "São Paulo",
        "Austin",
        "Montréal",
        "Sakhir",
        "Lusail",
        "Yas Island",
        "Le Castellet",
    ],
    "high_altitude_special": [
        "Mexico City"
    ],
    "mixed_characteristics": [],
}


def get_track_type(track_name: str) -> Optional[str]:
    track_name_clean = track_name.strip()

    for track_type, tracks in TRACK_CLUSTERS.items():
        for track in tracks:
            if (
                track.lower() in track_name_clean.lower()
                or track_name_clean.lower() in track.lower()
            ):
                return track_type

    logger.warning(f"Track '{track_name}' not found, defaulting to 'medium_downforce_balanced'")
    return "medium_downforce_balanced"


def add_track_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "track_name" not in df.columns:
        logger.error("DataFrame missing 'track_name' column")
        return df

    df["track_type"] = df["track_name"].apply(get_track_type)

    for track_type in TRACK_CLUSTERS.keys():
        df[f"track_type_{track_type}"] = (df["track_type"] == track_type).astype(int)

    logger.info(f"Added track type classification: {df['track_type'].nunique()} unique types")

    return df


def calculate_track_type_performance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "track_type" not in df.columns:
        df = add_track_type(df)

    df = df.sort_values(["driver", "year", "round"])

    for track_type in TRACK_CLUSTERS.keys():
        track_mask = df["track_type"] == track_type

        df[f"avg_finish_{track_type}"] = (
            df[track_mask]
            .groupby("driver")["finish_position"]
            .transform(lambda x: x.expanding().mean())
        )

    df["avg_finish_this_track_type"] = df.apply(
        lambda row: (
            row[f"avg_finish_{row['track_type']}"]
            if f"avg_finish_{row['track_type']}" in row.index
            else 10.0
        ),
        axis=1,
    )

    df["avg_finish_this_track_type"] = df["avg_finish_this_track_type"].fillna(10.0)

    return df
