"""
Ensemble models for F1 race prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import joblib

from src.utils.logging import get_logger
from src.utils.config import RANDOM_STATE

logger = get_logger(__name__)


class F1EnsembleModel:
    def __init__(
        self, random_state: int = RANDOM_STATE, use_optimized_params: bool = True
    ):
        self.random_state = random_state
        self.models = {}

        if use_optimized_params:
            self.models["random_forest"] = RandomForestRegressor(
                n_estimators=834,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            )

            self.models["gradient_boosting"] = GradientBoostingRegressor(
                n_estimators=2000,
                learning_rate=0.01,
                max_depth=3,
                random_state=random_state,
                verbose=0,
            )

            self.models["svr"] = SVR(kernel="rbf", C=1000, gamma="scale", verbose=False)
        else:
            self.models["random_forest"] = RandomForestRegressor(
                random_state=random_state, n_jobs=-1
            )

            self.models["gradient_boosting"] = GradientBoostingRegressor(
                random_state=random_state
            )

            self.models["svr"] = SVR()

        self.feature_names = None
        self.best_model_name = None
        self.training_history = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series, model_name: str = "random_forest"
    ) -> "F1EnsembleModel":
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from {list(self.models.keys())}"
            )

        self.feature_names = list(X.columns)
        self.models[model_name].fit(X, y)

        return self

    def predict(self, X: pd.DataFrame, model_name: str = "random_forest") -> np.ndarray:
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        return self.models[model_name].predict(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "random_forest",
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        y_pred = self.predict(X, model_name)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        spearman_rho, spearman_p = spearmanr(y, y_pred)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman_rho": spearman_rho,
            "spearman_p_value": spearman_p,
        }

        if return_predictions:
            metrics["predictions"] = y_pred

        return metrics

    def get_feature_importance(
        self, model_name: str = "random_forest", top_n: int = 20
    ) -> pd.DataFrame:
        model = self.models[model_name]

        if not hasattr(model, "feature_importances_"):
            logger.warning(f"{model_name} does not support feature importance")
            return pd.DataFrame()

        if self.feature_names is None:
            logger.error("Model has not been trained yet")
            return pd.DataFrame()

        importance_df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        return importance_df


def leave_one_year_out_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "finish_position",
    years: Optional[List[int]] = None,
    models_to_test: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    if years is None:
        years = sorted(df["year"].unique())

    if models_to_test is None:
        models_to_test = ["random_forest", "gradient_boosting", "svr"]

    logger.info(
        f"LOYO-CV: {len(years)} years, {len(models_to_test)} models, {len(feature_cols)} features"
    )

    results = {model_name: {} for model_name in models_to_test}

    for test_year in years:
        logger.info(f"\nTest year: {test_year}")

        train_data = df[df["year"] != test_year].copy()
        test_data = df[df["year"] == test_year].copy()

        train_data = train_data.dropna(subset=[target_col])
        test_data = test_data.dropna(subset=[target_col])

        logger.info(
            f"  Train: {len(train_data)} samples, Test: {len(test_data)} samples"
        )

        X_train = train_data[feature_cols].copy()
        y_train = train_data[target_col].copy()
        X_test = test_data[feature_cols].copy()
        y_test = test_data[target_col].copy()

        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())

        for model_name in models_to_test:
            ensemble = F1EnsembleModel()
            ensemble.fit(X_train, y_train, model_name=model_name)

            metrics = ensemble.evaluate(
                X_test, y_test, model_name=model_name, return_predictions=True
            )

            results[model_name][test_year] = {
                "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
                "predictions": metrics["predictions"],
                "actuals": y_test.values,
                "test_data": test_data,
                "model": ensemble,
            }

            logger.info(
                f"  {model_name}: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}, Spearman={metrics['spearman_rho']:.3f}"
            )

    logger.info("\nAggregate results:")

    for model_name in models_to_test:
        all_metrics = [results[model_name][year]["metrics"] for year in years]

        avg_rmse = np.mean([m["rmse"] for m in all_metrics])
        avg_mae = np.mean([m["mae"] for m in all_metrics])
        avg_r2 = np.mean([m["r2"] for m in all_metrics])
        avg_spearman = np.mean([m["spearman_rho"] for m in all_metrics])

        logger.info(
            f"  {model_name}: RMSE={avg_rmse:.3f}, MAE={avg_mae:.3f}, R²={avg_r2:.3f}, Spearman={avg_spearman:.3f}"
        )

        results[model_name]["aggregate"] = {
            "avg_rmse": avg_rmse,
            "avg_mae": avg_mae,
            "avg_r2": avg_r2,
            "avg_spearman_rho": avg_spearman,
        }

    best_model = min(models_to_test, key=lambda m: results[m]["aggregate"]["avg_rmse"])

    logger.info(f"\nBest model: {best_model}")

    return results


def train_final_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "finish_position",
    model_name: str = "random_forest",
    save_path: Optional[str] = None,
) -> F1EnsembleModel:
    logger.info(f"Training final {model_name} model")

    df_clean = df.dropna(subset=[target_col])
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()
    X = X.fillna(X.median())

    logger.info(
        f"  {len(X)} samples, {len(feature_cols)} features, years: {sorted(df_clean['year'].unique())}"
    )

    ensemble = F1EnsembleModel()
    ensemble.fit(X, y, model_name=model_name)

    metrics = ensemble.evaluate(X, y, model_name=model_name)

    logger.info(
        f"  Training: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}, Spearman={metrics['spearman_rho']:.3f}"
    )

    importance_df = ensemble.get_feature_importance(model_name=model_name, top_n=10)
    logger.info("\nTop 10 features:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    if save_path:
        joblib.dump(ensemble, save_path)
        logger.info(f"\n✓ Saved to {save_path}")

    return ensemble
