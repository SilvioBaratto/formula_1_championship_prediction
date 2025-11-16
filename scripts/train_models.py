#!/usr/bin/env python3
"""
Train F1 prediction models using Leave-One-Year-Out Cross-Validation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import joblib
import pandas as pd
from datetime import datetime

from src.models.ensemble import leave_one_year_out_cv, train_final_model, F1EnsembleModel
from src.models.bayesian import (
    calculate_variance_decomposition,
    add_driver_constructor_features,
    analyze_driver_team_combinations
)
from src.utils.config import (
    PROCESSED_DATA_DIR,
    FEATURE_MATRIX_FILE,
    MODELS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_feature_matrix():
    feature_path = PROCESSED_DATA_DIR / FEATURE_MATRIX_FILE

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}")

    df = pd.read_csv(feature_path)
    logger.info(f"Loaded {len(df)} records from {df['year'].nunique()} seasons")
    return df


def perform_variance_analysis(df: pd.DataFrame):
    results = calculate_variance_decomposition(
        df,
        position_col='finish_position_clean',
        driver_col='driver',
        team_col='team'
    )

    if 2024 in df['year'].values:
        analyze_driver_team_combinations(df, year=2024)

    logger.info(f"Variance: Constructor {results['constructor_contribution_pct']:.1f}%, "
                f"Driver {results['driver_contribution_pct']:.1f}%")

    return results


def validate_features(df: pd.DataFrame, feature_cols: list):
    valid = [col for col in feature_cols if col in df.columns]
    missing = [col for col in feature_cols if col not in df.columns]

    if missing:
        logger.warning(f"Missing {len(missing)} features")

    logger.info(f"Using {len(valid)} features for training")
    return valid


def run_loyo_cv(df: pd.DataFrame, feature_cols: list, years: list):
    logger.info(f"Running LOYO-CV on {years}")

    cv_results = leave_one_year_out_cv(
        df=df[df['year'].isin(years)],
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        years=years,
        models_to_test=['random_forest', 'gradient_boosting', 'svr']
    )

    return cv_results


def select_best_model(cv_results: dict):
    rankings = []

    for model_name, results in cv_results.items():
        if 'aggregate' not in results:
            continue

        agg = results['aggregate']
        rankings.append({
            'model': model_name,
            'spearman': agg['avg_spearman_rho'],
            'rmse': agg['avg_rmse'],
            'r2': agg['avg_r2']
        })

    rankings.sort(key=lambda x: (-x['spearman'], x['rmse']))

    logger.info("\nModel Rankings:")
    for rank, m in enumerate(rankings, 1):
        logger.info(f"  {rank}. {m['model'].upper()}: "
                   f"Spearman={m['spearman']:.3f}, RMSE={m['rmse']:.3f}, R²={m['r2']:.3f}")

    best = rankings[0]['model']
    logger.info(f"\nSelected: {best.upper()}")

    return best


def train_and_save_model(df: pd.DataFrame, feature_cols: list, model_name: str, training_years: list):
    training_data = df[df['year'].isin(training_years)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"f1_model_{model_name}_{timestamp}.joblib"

    final_model = train_final_model(
        df=training_data,
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        model_name=model_name,
        save_path=str(model_path)
    )

    latest_path = MODELS_DIR / f"f1_model_{model_name}_latest.joblib"
    joblib.dump(final_model, latest_path)
    logger.info(f"Saved to {latest_path}")

    return final_model


def save_metadata(cv_results: dict, variance_results: dict, best_model: str,
                 feature_cols: list, training_years: list):
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model,
        'training_years': training_years,
        'n_features': len(feature_cols),
        'variance_decomposition': {
            'constructor_pct': variance_results['constructor_contribution_pct'],
            'driver_pct': variance_results['driver_contribution_pct']
        },
        'cv_results': {
            name: res['aggregate']
            for name, res in cv_results.items()
            if 'aggregate' in res
        }
    }

    metadata_path = MODELS_DIR / f"training_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Metadata saved to {metadata_path}")


def main():
    logger.info("Training F1 prediction models")

    try:
        df = load_feature_matrix()

        df = add_driver_constructor_features(df)
        variance_results = perform_variance_analysis(df)

        training_years = [2022, 2023, 2024]
        available_years = sorted(df['year'].unique())
        training_years = [y for y in training_years if y in available_years]

        valid_features = validate_features(df, FEATURE_COLUMNS)

        bayesian_features = [
            'driver_avg_position_historical',
            'team_avg_position_historical',
            'driver_skill_vs_team',
            'team_strength'
        ]

        for feat in bayesian_features:
            if feat in df.columns and feat not in valid_features:
                valid_features.append(feat)

        if not valid_features:
            logger.error("No valid features found")
            return 1

        logger.info(f"Training with {len(valid_features)} features on {training_years}")

        cv_results = run_loyo_cv(df, valid_features, training_years)
        best_model = select_best_model(cv_results)

        train_and_save_model(df, valid_features, best_model, training_years)
        save_metadata(cv_results, variance_results, best_model, valid_features, training_years)

        logger.info("Training complete")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
