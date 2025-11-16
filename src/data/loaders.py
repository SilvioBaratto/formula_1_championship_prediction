"""
Data loading functions for F1 race data using FastF1 API.
"""

import time
from functools import wraps
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

try:
    import fastf1
    from fastf1.core import Session
except ImportError:
    raise ImportError("FastF1 library not installed. Run: pip install fastf1")

from src.utils.config import (
    CACHE_DIR,
    API_RETRY_INITIAL_WAIT,
    API_RETRY_MAX_WAIT,
    API_RETRY_MAX_ATTEMPTS,
    RACE_LOAD_DELAY,
    YEARS,
    RAW_DATA_DIR,
    RAW_RACE_RESULTS_FILE,
    RAW_QUALIFYING_RESULTS_FILE,
    TEAM_NAME_MAPPINGS,
)
from src.utils.logging import get_logger, configure_fastf1_logging

logger = get_logger(__name__)

fastf1.Cache.enable_cache(str(CACHE_DIR))
configure_fastf1_logging("INFO")


def retry_with_exponential_backoff(max_attempts: int = API_RETRY_MAX_ATTEMPTS):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = API_RETRY_INITIAL_WAIT
            last_exception = Exception("Unknown error")

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    if (
                        "429" in error_msg
                        or "rate limit" in error_msg
                        or "too many requests" in error_msg
                    ):
                        if attempt < max_attempts:
                            logger.warning(
                                f"Rate limit hit (attempt {attempt}/{max_attempts}), waiting {wait_time}s"
                            )
                            time.sleep(wait_time)
                            wait_time = min(wait_time * 2, API_RETRY_MAX_WAIT)
                        else:
                            logger.error(f"Max retries reached: {e}")
                            raise
                    else:
                        logger.error(f"Error in {func.__name__}: {e}")
                        raise

            raise last_exception

        return wrapper

    return decorator


def standardize_team_name(team_name: str) -> str:
    if pd.isna(team_name):
        return team_name

    team_name = str(team_name).strip()

    for standard_name, variations in TEAM_NAME_MAPPINGS.items():
        if team_name in variations or team_name == standard_name:
            return standard_name

    return team_name


@retry_with_exponential_backoff()
def load_session(year: int, race: int, session_type: str = "R") -> Session:
    logger.info(f"Loading {year} Round {race} - Session: {session_type}")

    session = fastf1.get_session(year, race, session_type)
    session.load(laps=False, telemetry=False, weather=True, messages=True)

    return session


def extract_safety_car_data(session: Session) -> Dict[str, Any]:
    try:
        if (
            not hasattr(session, "race_control_messages")
            or session.race_control_messages is None
        ):
            return {
                "has_safety_car": False,
                "has_vsc": False,
                "sc_count": 0,
                "vsc_count": 0,
            }

        messages = session.race_control_messages

        if len(messages) == 0 or "Message" not in messages.columns:
            return {
                "has_safety_car": False,
                "has_vsc": False,
                "sc_count": 0,
                "vsc_count": 0,
            }

        sc_messages = messages[
            messages["Message"].str.contains("SAFETY CAR", case=False, na=False)
        ]
        vsc_messages = messages[
            messages["Message"].str.contains("VIRTUAL SAFETY CAR", case=False, na=False)
        ]

        sc_deployed = sc_messages[
            sc_messages["Message"].str.contains(
                "DEPLOYED|ON TRACK|IN THIS LAP", case=False, na=False
            )
        ]
        sc_overtake = sc_messages[
            sc_messages["Message"].str.contains("MAY OVERTAKE", case=False, na=False)
        ]
        vsc_deployed = vsc_messages[
            vsc_messages["Message"].str.contains("DEPLOYED", case=False, na=False)
        ]

        has_sc = len(sc_deployed) > 0 or len(sc_overtake) > 0
        has_vsc = len(vsc_deployed) > 0

        return {
            "has_safety_car": has_sc,
            "has_vsc": has_vsc,
            "sc_count": max(len(sc_deployed), len(sc_overtake)),
            "vsc_count": len(vsc_deployed),
        }

    except Exception as e:
        logger.debug(f"Error extracting safety car data: {e}")
        return {
            "has_safety_car": False,
            "has_vsc": False,
            "sc_count": 0,
            "vsc_count": 0,
        }


def extract_weather_data(session: Session) -> Dict[str, Any]:
    try:
        weather = session.weather_data

        if weather is None or len(weather) == 0:
            return {
                "air_temp": np.nan,
                "track_temp": np.nan,
                "humidity": np.nan,
                "pressure": np.nan,
                "rainfall": False,
                "wind_speed": np.nan,
            }

        return {
            "air_temp": (
                weather["AirTemp"].mean() if "AirTemp" in weather.columns else np.nan
            ),
            "track_temp": (
                weather["TrackTemp"].mean()
                if "TrackTemp" in weather.columns
                else np.nan
            ),
            "humidity": (
                weather["Humidity"].mean() if "Humidity" in weather.columns else np.nan
            ),
            "pressure": (
                weather["Pressure"].mean() if "Pressure" in weather.columns else np.nan
            ),
            "rainfall": (
                weather["Rainfall"].any() if "Rainfall" in weather.columns else False
            ),
            "wind_speed": (
                weather["WindSpeed"].mean()
                if "WindSpeed" in weather.columns
                else np.nan
            ),
        }

    except Exception as e:
        logger.warning(f"Could not extract weather data: {e}")
        return {
            "air_temp": np.nan,
            "track_temp": np.nan,
            "humidity": np.nan,
            "pressure": np.nan,
            "rainfall": False,
            "wind_speed": np.nan,
        }


def extract_race_results(session: Session, year: int, round_num: int) -> pd.DataFrame:
    results = session.results

    if results is None or len(results) == 0:
        logger.warning(f"No results found for {year} Round {round_num}")
        return pd.DataFrame()

    weather_data = extract_weather_data(session)
    safety_car_data = extract_safety_car_data(session)

    race_data = []

    for _, row in results.iterrows():
        race_record = {
            "year": year,
            "round": round_num,
            "race_name": session.event["EventName"],
            "track_name": session.event["Location"],
            "date": session.event["EventDate"],
            "driver": row.get("Abbreviation", ""),
            "driver_number": row.get("DriverNumber", ""),
            "driver_name": row.get("BroadcastName", ""),
            "team": standardize_team_name(row.get("TeamName", "")),
            "team_color": row.get("TeamColor", ""),
            "grid_position": row.get("GridPosition", np.nan),
            "finish_position": row.get("Position", np.nan),
            "points": row.get("Points", 0.0),
            "status": row.get("Status", ""),
            "race_time": row.get("Time", pd.NaT),
            **weather_data,
            **safety_car_data,
        }

        race_data.append(race_record)

    df = pd.DataFrame(race_data)
    logger.info(f"Extracted {len(df)} driver results for {year} Round {round_num}")

    return df


def extract_qualifying_results(
    session: Session, year: int, round_num: int
) -> pd.DataFrame:
    results = session.results

    if results is None or len(results) == 0:
        logger.warning(f"No qualifying results for {year} Round {round_num}")
        return pd.DataFrame()

    qualifying_data = []

    for _, row in results.iterrows():
        qual_record = {
            "year": year,
            "round": round_num,
            "driver": row.get("Abbreviation", ""),
            "team": standardize_team_name(row.get("TeamName", "")),
            "q1_time": row.get("Q1", pd.NaT),
            "q2_time": row.get("Q2", pd.NaT),
            "q3_time": row.get("Q3", pd.NaT),
            "position": row.get("Position", np.nan),
        }

        qualifying_data.append(qual_record)

    df = pd.DataFrame(qualifying_data)
    logger.info(f"Extracted {len(df)} qualifying results for {year} Round {round_num}")

    return df


@retry_with_exponential_backoff()
def get_event_schedule_with_retry(year: int):
    logger.info(f"Fetching event schedule for {year}...")
    return fastf1.get_event_schedule(year)


def load_season_races(year: int, session_type: str = "R") -> pd.DataFrame:
    logger.info(f"Loading {year} season - Session type: {session_type}")

    try:
        schedule = get_event_schedule_with_retry(year)
    except Exception as e:
        logger.error(f"Failed to load schedule for {year}: {e}")
        return pd.DataFrame()

    schedule = schedule[schedule["EventFormat"] != "testing"]

    all_data = []
    successful = 0
    failed = 0

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]
        event_date = event.get("EventDate", "Unknown date")

        logger.info(f"Attempting: Round {round_num} - {event_name} ({event_date})")

        try:
            session = load_session(year, round_num, session_type)

            if session_type == "R":
                df = extract_race_results(session, year, round_num)
            elif session_type == "Q":
                df = extract_qualifying_results(session, year, round_num)
            else:
                df = pd.DataFrame()

            if not df.empty:
                all_data.append(df)
                successful += 1
                logger.info(f"SUCCESS: Loaded {len(df)} records from {event_name}")
            else:
                failed += 1
                logger.warning(f"FAILED: No data for {event_name}")

            time.sleep(RACE_LOAD_DELAY)

        except Exception as e:
            failed += 1
            logger.error(
                f"EXCEPTION: Failed to load {year} Round {round_num} ({event_name}): {e}"
            )
            continue

    logger.info(
        f"\n{year} Summary: {successful} successful, {failed} failed out of {len(schedule)} races"
    )

    if not all_data:
        logger.warning(f"No data collected for {year} {session_type}")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {year} season: {len(df)} records from {len(all_data)} events")

    return df


def load_all_historical_data(
    years: Optional[List[int]] = None,
    save_to_csv: bool = True,
    load_qualifying: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if years is None:
        years = YEARS

    logger.info(f"Loading historical data for {years}")

    all_race_data = []
    all_qualifying_data = []

    for year in years:
        logger.info(f"Processing {year} season")

        race_df = load_season_races(year, session_type="R")
        if not race_df.empty:
            all_race_data.append(race_df)
            logger.info(f"Loaded {len(race_df)} race records for {year}")

        if load_qualifying:
            qual_df = load_season_races(year, session_type="Q")
            if not qual_df.empty:
                all_qualifying_data.append(qual_df)
                logger.info(f"Loaded {len(qual_df)} qualifying records for {year}")

        if year != years[-1]:
            time.sleep(RACE_LOAD_DELAY)

    race_data = (
        pd.concat(all_race_data, ignore_index=True) if all_race_data else pd.DataFrame()
    )
    qualifying_data = (
        pd.concat(all_qualifying_data, ignore_index=True)
        if all_qualifying_data
        else pd.DataFrame()
    )

    logger.info(
        f"Total: {len(race_data)} race records, {len(qualifying_data)} qualifying records"
    )

    if save_to_csv and not race_data.empty:
        race_csv_path = RAW_DATA_DIR / RAW_RACE_RESULTS_FILE
        race_data.to_csv(race_csv_path, index=False)
        logger.info(f"Saved race data to {race_csv_path}")

    if save_to_csv and not qualifying_data.empty:
        qual_csv_path = RAW_DATA_DIR / RAW_QUALIFYING_RESULTS_FILE
        qualifying_data.to_csv(qual_csv_path, index=False)
        logger.info(f"Saved qualifying data to {qual_csv_path}")

    return race_data, qualifying_data


def get_current_season_schedule(year: int = 2025) -> pd.DataFrame:
    logger.info(f"Fetching {year} season schedule")

    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule["EventFormat"] != "testing"]

    logger.info(f"Found {len(schedule)} race events for {year}")

    return schedule


def get_completed_races_2025(use_cache: bool = True) -> pd.DataFrame:
    from src.utils.config import PROCESSED_DATA_DIR, COMPLETED_2025_RACES_FILE

    cache_path = PROCESSED_DATA_DIR / COMPLETED_2025_RACES_FILE

    if use_cache and cache_path.exists():
        race_data = pd.read_csv(cache_path)
        logger.info(f"✓ Loaded {len(race_data)} 2025 race records from cache")
        return race_data

    race_data = load_season_races(2025, session_type="R")

    if not race_data.empty:
        race_data.to_csv(cache_path, index=False)
        logger.info(f"✓ Loaded {len(race_data)} 2025 race records (saved to cache)")

    return race_data


def get_championship_standings(race_data: pd.DataFrame, year: int) -> pd.DataFrame:
    year_data = race_data[race_data["year"] == year].copy()

    if year_data.empty:
        logger.warning(f"No race data found for {year}")
        return pd.DataFrame()

    standings = (
        year_data.groupby(["driver", "driver_name", "team"])["points"]
        .sum()
        .reset_index()
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )

    standings["position"] = standings.index + 1

    logger.info(f"{year} Championship Standings - Top 5:")
    logger.info("\n" + standings.head().to_string(index=False))

    return standings


if __name__ == "__main__":
    logger.info("Testing data loader")

    race_data, qualifying_data = load_all_historical_data(years=YEARS, save_to_csv=True)

    logger.info(f"\nRace data: {race_data.shape}")
    logger.info(f"Qualifying data: {qualifying_data.shape}")

    if not race_data.empty:
        get_championship_standings(race_data, 2025)
