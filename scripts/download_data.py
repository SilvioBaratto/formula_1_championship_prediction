#!/usr/bin/env python3
"""
Download historical F1 data from FastF1 API.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_all_historical_data, get_championship_standings
from src.utils.config import YEARS, RAW_DATA_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info(f"Downloading data for {YEARS}")

    try:
        race_data, qualifying_data = load_all_historical_data(
            years=YEARS,
            save_to_csv=True,
            load_qualifying=False
        )

        if race_data.empty:
            logger.error("No data downloaded")
            return 1

        logger.info(f"Downloaded {len(race_data)} race records")

        races_per_year = race_data.groupby('year')['round'].nunique()
        for year, count in races_per_year.items():
            logger.info(f"  {year}: {count} races")

        if 2025 in race_data['year'].values:
            standings = get_championship_standings(race_data, 2025)
            logger.info(f"\n2025 Standings:\n{standings.to_string(index=False)}")

        logger.info(f"\nSaved to {RAW_DATA_DIR}")
        return 0

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
