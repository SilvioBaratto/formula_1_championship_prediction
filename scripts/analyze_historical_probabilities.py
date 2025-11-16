#!/usr/bin/env python3
"""
Historical probability analysis for F1 Monte Carlo simulation.
Extracts DNF rates, weather impact, safety car frequencies, and driver consistency.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.features.track_clustering import get_track_type
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw_race_data() -> pd.DataFrame:
    data_path = RAW_DATA_DIR / "race_results_raw.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Race data not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} results from {df['year'].nunique()} seasons")
    return df


def analyze_dnf_rates(df: pd.DataFrame) -> Dict:
    completed_statuses = ['Finished', 'Lapped', '+1 Lap', '+2 Laps', '+3 Laps', '+4 Laps']
    df['is_dnf'] = ~df['status'].isin(completed_statuses)

    overall_rate = df['is_dnf'].mean()

    driver_rates = df.groupby('driver').agg(
        dnf_count=('is_dnf', 'sum'),
        total=('is_dnf', 'count'),
        rate=('is_dnf', 'mean')
    )
    driver_rates = driver_rates[driver_rates['total'] >= 5]

    team_rates = df.groupby('team').agg(
        dnf_count=('is_dnf', 'sum'),
        total=('is_dnf', 'count'),
        rate=('is_dnf', 'mean')
    )
    team_rates = team_rates[team_rates['total'] >= 10]

    df['track_type'] = df['track_name'].apply(get_track_type)
    track_rates = df.groupby('track_type')['is_dnf'].mean()

    logger.info(f"DNF rate: {overall_rate*100:.1f}%")

    return {
        'overall_dnf_rate': overall_rate,
        'driver_dnf_rates': driver_rates['rate'].to_dict(),
        'team_dnf_rates': team_rates['rate'].to_dict(),
        'track_type_dnf_rates': track_rates.to_dict(),
    }


def analyze_weather_impact(df: pd.DataFrame) -> Dict:
    df['is_wet'] = df['rainfall'] == True

    wet_races = df.groupby(['year', 'round'])['is_wet'].first()
    wet_prob = wet_races.mean()

    df['track_type'] = df['track_name'].apply(get_track_type)
    track_wet = df.groupby(['track_type', 'year', 'round'])['is_wet'].first().reset_index()
    track_wet_prob = track_wet.groupby('track_type')['is_wet'].mean()

    wet_var = df[df['is_wet']].groupby('driver')['finish_position'].std().mean()
    dry_var = df[~df['is_wet']].groupby('driver')['finish_position'].std().mean()
    chaos_mult = wet_var / dry_var

    driver_wet_perf = []
    for driver in df['driver'].unique():
        d = df[df['driver'] == driver]
        if len(d) < 10:
            continue

        wet = d[d['is_wet']]
        dry = d[~d['is_wet']]

        if len(wet) >= 2:
            advantage = dry['finish_position'].mean() - wet['finish_position'].mean()
            driver_wet_perf.append({'driver': driver, 'advantage': advantage})

    wet_adv = pd.DataFrame(driver_wet_perf)

    logger.info(f"Wet race probability: {wet_prob*100:.1f}% (chaos: {chaos_mult:.2f}x)")

    return {
        'overall_wet_probability': wet_prob,
        'track_type_wet_probability': track_wet_prob.to_dict(),
        'wet_chaos_multiplier': chaos_mult,
        'driver_wet_advantage': dict(zip(wet_adv['driver'], wet_adv['advantage']))
    }


def analyze_safety_car_indicators(df: pd.DataFrame) -> Dict:
    if 'has_safety_car' not in df.columns:
        logger.warning("Safety car data not available, using fallback")
        return _fallback_sc_analysis(df)

    races = df.groupby(['year', 'round', 'race_name', 'track_name']).agg({
        'has_safety_car': 'first',
        'has_vsc': 'first',
        'sc_count': 'first',
        'vsc_count': 'first',
    }).reset_index()

    races['track_type'] = races['track_name'].apply(get_track_type)
    races['pos_change'] = df.groupby(['year', 'round']).apply(
        lambda x: (x['finish_position'] - x['grid_position']).abs().mean()
    ).values

    overall_sc = races['has_safety_car'].mean()
    track_type_sc = races.groupby('track_type')['has_safety_car'].mean()
    track_specific_sc = races.groupby('track_name')['has_safety_car'].mean()
    volatility = races.groupby('track_type')['pos_change'].mean()

    logger.info(f"Safety car probability: {overall_sc*100:.1f}%")

    return {
        'overall_sc_probability': overall_sc,
        'overall_vsc_probability': races['has_vsc'].mean(),
        'track_type_sc_probability': track_type_sc.to_dict(),
        'track_specific_sc_probability': track_specific_sc.to_dict(),
        'track_type_position_volatility': volatility.to_dict()
    }


def _fallback_sc_analysis(df: pd.DataFrame) -> Dict:
    races = df.groupby(['year', 'round', 'track_name']).agg({
        'status': lambda x: (x != 'Finished').sum(),
        'finish_position': lambda x: x.values,
        'grid_position': lambda x: x.values
    }).reset_index()

    races['track_type'] = races['track_name'].apply(get_track_type)
    races['likely_sc'] = races['status'] >= 3
    races['pos_change'] = races.apply(
        lambda r: np.abs(r['finish_position'] - r['grid_position']).mean(), axis=1
    )

    return {
        'overall_sc_probability': races['likely_sc'].mean(),
        'track_type_sc_probability': races.groupby('track_type')['likely_sc'].mean().to_dict(),
        'track_type_position_volatility': races.groupby('track_type')['pos_change'].mean().to_dict()
    }


def analyze_driver_consistency(df: pd.DataFrame) -> Dict:
    consistency = df.groupby('driver').agg(
        avg_pos=('finish_position', 'mean'),
        std_pos=('finish_position', 'std'),
        races=('finish_position', 'count')
    )
    consistency = consistency[consistency['races'] >= 10]

    return {
        'driver_variance': consistency['std_pos'].to_dict()
    }


def save_results(results: Dict):
    output_path = PROCESSED_DATA_DIR / "historical_probabilities.json"

    def convert_numpy(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    converted: Dict = convert_numpy(results)

    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)

    logger.info(f"Saved to {output_path}")


def main():
    logger.info("Analyzing historical race probabilities")

    df = load_raw_race_data()

    results = {
        'dnf': analyze_dnf_rates(df),
        'weather': analyze_weather_impact(df),
        'safety_car': analyze_safety_car_indicators(df),
        'consistency': analyze_driver_consistency(df)
    }

    save_results(results)
    logger.info("Analysis complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
