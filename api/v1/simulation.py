"""
F1 Championship simulation endpoint.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import get_current_season_schedule, get_completed_races_2025
from src.features.engineering import engineer_prediction_features
from src.features.track_clustering import get_track_type
from src.models.monte_carlo import run_monte_carlo_prediction
from src.utils.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_MATRIX_FILE,
    DRIVER_TEAM_2025,
    PREDICTION_YEAR,
    REMAINING_RACES_2025,
)

router = APIRouter(prefix="/api/v1", tags=["simulation"])


class DriverStanding(BaseModel):
    driver: str
    expected_position: float
    prob_win: float
    prob_podium: float
    expected_points: float
    current_points: float


class SimulationResponse(BaseModel):
    championship_predictions: List[DriverStanding]
    winner_probabilities: List[dict]
    simulation_params: dict


def load_trained_model():
    model_files = list(MODELS_DIR.glob("f1_model_*_latest.joblib"))
    if not model_files:
        raise FileNotFoundError("No trained model found")

    model_path = model_files[0]
    model_type = model_path.stem.replace('f1_model_', '').replace('_latest', '')

    model = joblib.load(model_path)
    return model, model_path, model_type


def load_feature_matrix():
    feature_path = PROCESSED_DATA_DIR / FEATURE_MATRIX_FILE
    if not feature_path.exists():
        raise FileNotFoundError("Feature matrix not found")

    df = pd.read_csv(feature_path)
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

    return schedule, remaining_races_info


def get_completed_2025_data():
    try:
        completed_races = get_completed_races_2025(use_cache=True)
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

            return completed_races, standings
        else:
            return pd.DataFrame(), pd.DataFrame()

    except Exception as e:
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
    model,
    prediction_features: pd.DataFrame,
    feature_cols: List[str],
    model_type: str
) -> pd.DataFrame:
    available_features = [col for col in feature_cols if col in prediction_features.columns]

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
    else:
        prediction_features['ml_predicted_position'] = 10.0

    return prediction_features


def run_hybrid_monte_carlo(
    ml_predictions: pd.DataFrame,
    remaining_races_info: List[dict],
    current_standings: pd.DataFrame,
    historical_data: pd.DataFrame,
    n_simulations: int = 10000
):
    remaining_races_for_mc = []

    for race_info in remaining_races_info:
        race_round = race_info['round']
        race_preds = ml_predictions[ml_predictions['round'] == race_round]

        ml_predictions_dict = dict(
            zip(race_preds['driver'], race_preds['ml_predicted_position'])
        )

        remaining_races_for_mc.append({
            'round': race_round,
            'track_name': race_info['location'],
            'track_type': race_info['track_type'],
            'ml_predictions': ml_predictions_dict
        })

    analysis_df, mc_results = run_monte_carlo_prediction(
        historical_data=historical_data,
        remaining_races=remaining_races_for_mc,
        current_standings=current_standings,
        driver_teams=DRIVER_TEAM_2025,
        n_simulations=n_simulations,
        use_ml_baseline=True
    )

    return analysis_df, mc_results


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_championship(n_simulations: int = 10000):
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
            n_simulations=n_simulations
        )

        championship_predictions = []
        for _, row in mc_analysis.iterrows():
            championship_predictions.append(
                DriverStanding(
                    driver=row['driver'],
                    expected_position=round(row['expected_position'], 2),
                    prob_win=round(row['prob_win'] * 100, 2),
                    prob_podium=round(row['prob_podium'] * 100, 2),
                    expected_points=round(row['expected_points'], 1),
                    current_points=round(row['current_points'], 1)
                )
            )

        winner_probabilities = [
            {
                "driver": row['driver'],
                "probability": round(row['prob_win'] * 100, 2)
            }
            for _, row in mc_analysis.nlargest(10, 'prob_win').iterrows()
        ]

        return SimulationResponse(
            championship_predictions=championship_predictions,
            winner_probabilities=winner_probabilities,
            simulation_params={
                "n_simulations": n_simulations,
                "remaining_races": len(remaining_races_info),
                "completed_races": len(completed_races) // 20 if not completed_races.empty else 0,
                "model_type": f"hybrid_{model_type}_monte_carlo"
            }
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
