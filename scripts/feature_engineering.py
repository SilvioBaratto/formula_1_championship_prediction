#!/usr/bin/env python3
"""
Apply feature engineering to preprocessed F1 data.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.features.engineering import engineer_features_pipeline
from src.utils.config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILE, FEATURE_MATRIX_FILE
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    preprocessed_path = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE

    if not preprocessed_path.exists():
        logger.error(f"Preprocessed data not found: {preprocessed_path}")
        return 1

    try:
        df = pd.read_csv(preprocessed_path)
        logger.info(f"Loaded {len(df)} records from {df['year'].nunique()} seasons")

        df_features = engineer_features_pipeline(df)

        feature_matrix_path = PROCESSED_DATA_DIR / FEATURE_MATRIX_FILE
        df_features.to_csv(feature_matrix_path, index=False)

        logger.info(f"Created {len(df_features.columns)} features ({len(df_features)} records)")

        missing = df_features.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) > 0:
            logger.warning("Features with missing values:")
            for col, count in missing.items():
                logger.warning(f"  {col}: {count} ({count/len(df_features)*100:.1f}%)")

        logger.info(f"Saved to {feature_matrix_path}")
        return 0

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
