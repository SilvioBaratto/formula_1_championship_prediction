"""
Configuration constants for F1 championship prediction.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".fastf1_cache"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2022, 2023, 2024, 2025]
CURRENT_YEAR = 2025
PREDICTION_YEAR = 2025
REMAINING_RACES_2025 = 3

API_RETRY_INITIAL_WAIT = 8
API_RETRY_MAX_WAIT = 512
API_RETRY_MAX_ATTEMPTS = 5
RACE_LOAD_DELAY = 5

RECENT_RACES_WINDOW = 3
FORM_WINDOW = 5
DNF_WINDOW = 10

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

POINTS_SYSTEM = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

DNF_KEYWORDS = [
    "Accident",
    "Collision",
    "Spun off",
    "Damage",
    "Brake",
    "Transmission",
    "Clutch",
    "Hydraulics",
    "Engine",
    "Gearbox",
    "Electrical",
    "Power Unit",
    "ERS",
    "Withdrew",
    "Did not start",
    "Retired",
    "Disqualified",
]

DNF_POSITION = 20

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FASTF1_LOG_LEVEL = "INFO"

SESSION_TYPES = {
    "race": "R",
    "qualifying": "Q",
    "sprint": "S",
    "sprint_qualifying": "SQ",
}

FEATURE_COLUMNS = [
    "grid_position",
    "grid_squared",
    "grid_log",
    "points_last_3_races",
    "avg_finish_last_5_races",
    "dnf_rate_last_10_races",
    "cumulative_points",
    "championship_position",
    "team_avg_position_season",
    "track_type_high_speed_low_downforce",
    "track_type_street_circuit_high_downforce",
    "track_type_high_downforce_technical",
    "track_type_medium_downforce_balanced",
    "track_type_high_altitude_special",
    "track_type_mixed_characteristics",
    "avg_finish_this_track_type",
    "air_temp",
    "track_temp",
    "humidity",
    "rainfall",
    "wind_speed",
    "gap_to_pole",
]

TARGET_COLUMN = "finish_position"

RAW_RACE_RESULTS_FILE = "race_results_raw.csv"
RAW_QUALIFYING_RESULTS_FILE = "qualifying_results_raw.csv"
RAW_LAP_TIMES_FILE = "lap_times_raw.csv"
PROCESSED_DATA_FILE = "processed_race_data.csv"
FEATURE_MATRIX_FILE = "feature_matrix.csv"
COMPLETED_2025_RACES_FILE = "completed_2025_races.csv"

DRIVER_TEAM_2025 = {
    "VER": "Red Bull",
    "LAW": "Red Bull",
    "TSU": "Racing Bulls",
    "HAD": "Racing Bulls",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "DOO": "Alpine",
    "COL": "Alpine",
    "OCO": "Haas",
    "BEA": "Haas",
    "ALB": "Williams",
    "SAI": "Williams",
    "HUL": "Sauber",
    "BOR": "Sauber",
}

TEAM_NAME_MAPPINGS = {
    "Racing Bulls": "RB",
    "RB": "RB",
    "Sauber": "Sauber",
    "Kick Sauber": "Sauber",
    "Alfa Romeo": "Sauber",
    "Red Bull": "Red Bull Racing",
    "Red Bull Racing": "Red Bull Racing",
    "Aston Martin": "Aston Martin",
    "Aston Martin Aramco F1 Team": "Aston Martin",
    "AlphaTauri": "RB",
    "Scuderia AlphaTauri": "RB",
}
