#!/usr/bin/env python3
"""
Preprocess raw F1 data for model training.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data.preprocessors import preprocess_pipeline, save_preprocessed_data
from src.utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_RACE_RESULTS_FILE,
    PROCESSED_DATA_FILE
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    raw_data_path = RAW_DATA_DIR / RAW_RACE_RESULTS_FILE

    if not raw_data_path.exists():
        logger.error(f"Raw data not found: {raw_data_path}")
        return 1

    try:
        df = pd.read_csv(raw_data_path)
        logger.info(f"Loaded {len(df)} raw records")

        df_processed = preprocess_pipeline(df)

        processed_path = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE
        save_preprocessed_data(df_processed, processed_path)

        logger.info(f"Processed {len(df_processed)} records ({len(df_processed.columns)} columns)")
        logger.info(f"Saved to {processed_path}")
        return 0

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
