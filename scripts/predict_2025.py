#!/usr/bin/env python3
"""
2025 F1 Championship Prediction using hybrid ML + Monte Carlo approach.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from typing import List, Tuple, Optional

from src.data.loaders import get_current_season_schedule, get_completed_races_2025
from src.features.engineering import engineer_prediction_features
from src.features.track_clustering import get_track_type
from src.models.ensemble import F1EnsembleModel
from src.models.monte_carlo import run_monte_carlo_prediction
from src.utils.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_MATRIX_FILE,
    DRIVER_TEAM_2025,
    PREDICTION_YEAR,
    REMAINING_RACES_2025,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_trained_model(model_name: Optional[str] = None):
    if model_name:
        model_path = MODELS_DIR / f"f1_model_{model_name}_latest.joblib"
        model_type = model_name
    else:
        model_files = list(MODELS_DIR.glob("f1_model_*_latest.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No trained models in {MODELS_DIR}")
        model_path = model_files[0]
        model_type = model_path.stem.replace('f1_model_', '').replace('_latest', '')

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading {model_type} model")
    return joblib.load(model_path), model_path, model_type


def load_feature_matrix():
    feature_path = PROCESSED_DATA_DIR / FEATURE_MATRIX_FILE

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}")

    df = pd.read_csv(feature_path)
    logger.info(f"Loaded {len(df)} historical records")
    return df


def get_2025_schedule_info():
    schedule = get_current_season_schedule(year=PREDICTION_YEAR)
    remaining_schedule = schedule.tail(REMAINING_RACES_2025)

    remaining_races_info = []
    for _, race in remaining_schedule.iterrows():
        remaining_races_info.append({
            'round': race['RoundNumber'],
            'name': race['EventName'],
            'location': race['Location'],
            'country': race['Country'],
            'track_type': get_track_type(race['Location'])
        })

    logger.info(f"Predicting {len(remaining_races_info)} remaining races")
    for race in remaining_races_info:
        logger.info(f"  Round {race['round']}: {race['name']}")

    return schedule, remaining_races_info


def get_completed_2025_data():
    try:
        completed_races = get_completed_races_2025()
        n_completed = len(completed_races) // 20 if len(completed_races) > 0 else 0

        if not completed_races.empty:
            standings = (
                completed_races.groupby(['driver', 'team'])['points']
                .sum()
                .reset_index()
                .sort_values('points', ascending=False)
                .reset_index(drop=True)
            )
            standings['position'] = standings.index + 1

            logger.info(f"\nCurrent standings ({n_completed} races):")
            logger.info("\n" + standings.head(10).to_string(index=False))

            return completed_races, standings
        else:
            logger.warning("No completed races found")
            return pd.DataFrame(), pd.DataFrame()

    except Exception as e:
        logger.warning(f"Could not load completed races: {e}")
        return pd.DataFrame(), pd.DataFrame()


def prepare_ml_prediction_features(
    remaining_races_info: List[dict],
    completed_races: pd.DataFrame,
    historical_data: pd.DataFrame
) -> pd.DataFrame:
    prediction_rows = []

    for race in remaining_races_info:
        for driver, team in DRIVER_TEAM_2025.items():
            prediction_rows.append({
                'year': PREDICTION_YEAR,
                'round': race['round'],
                'race_name': race['name'],
                'track_name': race['location'],
                'driver': driver,
                'team': team,
            })

    prediction_template = pd.DataFrame(prediction_rows)

    prediction_features = engineer_prediction_features(
        prediction_df=prediction_template,
        historical_df=historical_data,
        completed_2025_df=completed_races if not completed_races.empty else None
    )

    return prediction_features


def get_ml_predictions(
    model: F1EnsembleModel,
    prediction_features: pd.DataFrame,
    feature_cols: List[str],
    model_type: str
) -> pd.DataFrame:
    available_features = [col for col in feature_cols if col in prediction_features.columns]
    missing_features = [col for col in feature_cols if col not in prediction_features.columns]

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features")

    X_pred = prediction_features[available_features].copy() if available_features else pd.DataFrame()

    if not X_pred.empty:
        X_pred = X_pred.fillna(X_pred.median())
        raw_predictions = model.predict(X_pred, model_name=model_type)
        prediction_features['ml_predicted_position'] = raw_predictions

        for round_num in prediction_features['round'].unique():
            race_mask = prediction_features['round'] == round_num
            race_predictions = pd.Series(prediction_features.loc[race_mask, 'ml_predicted_position'])
            ranked_positions = race_predictions.rank(method='first').astype(int)
            prediction_features.loc[race_mask, 'ml_predicted_position_ranked'] = ranked_positions

        logger.info(f"Generated ML predictions for {len(prediction_features)} entries")
    else:
        logger.error("No features available for prediction")
        prediction_features['ml_predicted_position'] = 10.0

    return prediction_features


def run_hybrid_monte_carlo(
    ml_predictions: pd.DataFrame,
    remaining_races_info: List[dict],
    current_standings: pd.DataFrame,
    historical_data: pd.DataFrame,
    n_simulations: int = 10000
) -> Tuple[pd.DataFrame, dict]:
    mc_remaining_races = []

    for race in remaining_races_info:
        race_round = race['round']
        race_ml_preds = ml_predictions[ml_predictions['round'] == race_round]

        ml_pred_dict = dict(zip(
            race_ml_preds['driver'],
            race_ml_preds['ml_predicted_position']
        ))

        mc_remaining_races.append({
            'track_type': race['track_type'],
            'race_name': race['name'],
            'track_name': race['location'],
            'ml_predictions': ml_pred_dict
        })

    logger.info(f"Running {n_simulations} Monte Carlo simulations")

    mc_analysis, mc_results = run_monte_carlo_prediction(
        historical_data=historical_data,
        remaining_races=mc_remaining_races,
        current_standings=current_standings,
        driver_teams=DRIVER_TEAM_2025,
        n_simulations=n_simulations,
        use_ml_baseline=True
    )

    return mc_analysis, mc_results


def main():
    logger.info(f"{PREDICTION_YEAR} Championship Prediction (Hybrid ML + Monte Carlo)")

    try:
        model, model_path, model_type = load_trained_model()
        historical_data = load_feature_matrix()

        schedule, remaining_races_info = get_2025_schedule_info()
        completed_races, current_standings = get_completed_2025_data()

        prediction_features = prepare_ml_prediction_features(
            remaining_races_info, completed_races, historical_data
        )

        ml_predictions = get_ml_predictions(
            model=model,
            prediction_features=prediction_features,
            feature_cols=model.feature_names if hasattr(model, 'feature_names') else list(prediction_features.columns),
            model_type=model_type
        )

        mc_analysis, mc_results = run_hybrid_monte_carlo(
            ml_predictions=ml_predictions,
            remaining_races_info=remaining_races_info,
            current_standings=current_standings,
            historical_data=historical_data,
            n_simulations=10000
        )

        logger.info("Prediction complete")
        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
