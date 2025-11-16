"""
Monte Carlo simulation for F1 championship prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import POINTS_SYSTEM
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MonteCarloSimulator:
    def __init__(
        self,
        historical_data: pd.DataFrame,
        n_simulations: int = 10000,
        random_state: int = 42,
        use_ml_baseline: bool = False
    ):
        self.historical_data = historical_data
        self.n_simulations = n_simulations
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.use_ml_baseline = use_ml_baseline

        self.driver_distributions = {}
        self.team_distributions = {}
        self.driver_team_distributions = {}

        self.historical_probabilities = self._load_historical_probabilities()
        self._fit_distributions()

    def _load_historical_probabilities(self) -> Dict:
        import json
        from src.utils.config import PROCESSED_DATA_DIR

        prob_path = PROCESSED_DATA_DIR / "historical_probabilities.json"

        if prob_path.exists():
            with open(prob_path, 'r') as f:
                probabilities = json.load(f)
            logger.info("✓ Loaded historical probabilities")
            return probabilities
        else:
            logger.warning("No historical probabilities file - using defaults")
            return {
                'dnf': {'overall_dnf_rate': 0.35},
                'weather': {'overall_wet_probability': 0.25, 'wet_chaos_multiplier': 1.0},
                'safety_car': {'overall_sc_probability': 0.85}
            }

    def _fit_distributions(self):
        df = self.historical_data.copy()

        for driver in df['driver'].unique():
            driver_data = df[df['driver'] == driver]['finish_position'].dropna()

            if len(driver_data) >= 5:
                self.driver_distributions[driver] = {
                    'mean': driver_data.mean(),
                    'std': driver_data.std(),
                    'data': driver_data.values,
                    'n_races': len(driver_data)
                }

        for team in df['team'].unique():
            team_data = df[df['team'] == team]['finish_position'].dropna()

            if len(team_data) >= 10:
                self.team_distributions[team] = {
                    'mean': team_data.mean(),
                    'std': team_data.std(),
                    'data': team_data.values,
                    'n_races': len(team_data)
                }

        for (driver, team), group in df.groupby(['driver', 'team']):
            positions = group['finish_position'].dropna()

            if len(positions) >= 3:
                self.driver_team_distributions[(driver, team)] = {
                    'mean': positions.mean(),
                    'std': positions.std(),
                    'data': positions.values,
                    'n_races': len(positions)
                }

        logger.info(f"✓ Fitted distributions: {len(self.driver_distributions)} drivers, "
                   f"{len(self.team_distributions)} teams, "
                   f"{len(self.driver_team_distributions)} driver-team combos")

    def _get_performance_variance(self, driver: str, team: str) -> float:
        if (driver, team) in self.driver_team_distributions:
            return self.driver_team_distributions[(driver, team)]['std']
        elif driver in self.driver_distributions:
            return self.driver_distributions[driver]['std']
        elif team in self.team_distributions:
            return self.team_distributions[team]['std']
        return 3.0

    def _get_dnf_probability(self, driver: str, team: str, track_type: Optional[str] = None) -> float:
        dnf_data = self.historical_probabilities.get('dnf', {})

        driver_dnf = dnf_data.get('driver_dnf_rates', {}).get(driver, dnf_data.get('overall_dnf_rate', 0.35))
        team_dnf = dnf_data.get('team_dnf_rates', {}).get(team, dnf_data.get('overall_dnf_rate', 0.35))

        if track_type:
            track_dnf = dnf_data.get('track_type_dnf_rates', {}).get(track_type, dnf_data.get('overall_dnf_rate', 0.35))
            combined_dnf = (driver_dnf + team_dnf + track_dnf) / 3.0
        else:
            combined_dnf = (driver_dnf + team_dnf) / 2.0

        return combined_dnf

    def _check_dnf(self, driver: str, team: str, track_type: Optional[str] = None) -> bool:
        dnf_probability = self._get_dnf_probability(driver, team, track_type)
        return self.rng.random() < dnf_probability

    def _sample_weather_scenario(self, track_type: Optional[str] = None) -> Dict:
        weather_data = self.historical_probabilities.get('weather', {})

        if track_type:
            wet_prob = weather_data.get('track_type_wet_probability', {}).get(
                track_type,
                weather_data.get('overall_wet_probability', 0.25)
            )
        else:
            wet_prob = weather_data.get('overall_wet_probability', 0.25)

        is_wet = self.rng.random() < wet_prob

        if is_wet:
            chaos_multiplier = weather_data.get('wet_chaos_multiplier', 1.0)
            chaos_multiplier *= self.rng.uniform(1.2, 1.5)
        else:
            chaos_multiplier = 1.0

        return {
            'is_wet': is_wet,
            'chaos_multiplier': chaos_multiplier
        }

    def _apply_wet_weather_adjustment(
        self,
        driver: str,
        position: float,
        is_wet: bool
    ) -> float:
        if not is_wet:
            return position

        weather_data = self.historical_probabilities.get('weather', {})
        wet_advantage = weather_data.get('driver_wet_advantage', {}).get(driver, 0.0)

        adjusted_position = position - wet_advantage

        return adjusted_position

    def _check_safety_car(
        self,
        track_name: Optional[str] = None,
        track_type: Optional[str] = None
    ) -> bool:
        sc_data = self.historical_probabilities.get('safety_car', {})

        if track_name:
            track_specific_probs = sc_data.get('track_specific_sc_probability', {})
            if track_name in track_specific_probs:
                sc_prob = track_specific_probs[track_name]
                return self.rng.random() < sc_prob

        if track_type:
            sc_prob = sc_data.get('track_type_sc_probability', {}).get(
                track_type,
                sc_data.get('overall_sc_probability', 0.5)
            )
        else:
            sc_prob = sc_data.get('overall_sc_probability', 0.5)

        return self.rng.random() < sc_prob

    def _apply_safety_car_chaos(
        self,
        positions: List[float],
        track_type: Optional[str] = None
    ) -> List[float]:
        sc_data = self.historical_probabilities.get('safety_car', {})

        if track_type:
            volatility = sc_data.get('track_type_position_volatility', {}).get(
                track_type,
                3.5
            )
        else:
            volatility = 3.5

        noise = self.rng.normal(0, volatility * 0.5, len(positions))
        adjusted_positions = [pos + n for pos, n in zip(positions, noise)]

        return adjusted_positions

    def _sample_position(
        self,
        driver: str,
        team: str,
        ml_prediction: Optional[float] = None,
        track_type: Optional[str] = None
    ) -> int:
        if self.use_ml_baseline and ml_prediction is not None:
            variance = self._get_performance_variance(driver, team)
            position = self.rng.normal(ml_prediction, variance)

        else:
            if (driver, team) in self.driver_team_distributions:
                dist = self.driver_team_distributions[(driver, team)]
                sampled_position = self.rng.choice(dist['data'])
                noise = self.rng.normal(0, 0.5)
                position = sampled_position + noise

            elif driver in self.driver_distributions:
                dist = self.driver_distributions[driver]
                position = self.rng.normal(dist['mean'], dist['std'])

            elif team in self.team_distributions:
                dist = self.team_distributions[team]
                position = self.rng.normal(dist['mean'], dist['std'])

            else:
                position = self.rng.normal(10, 4)

        position = int(np.clip(np.round(position), 1, 20))

        return position

    def simulate_race(
        self,
        drivers: List[str],
        teams: Dict[str, str],
        ml_predictions: Optional[Dict[str, float]] = None,
        track_type: Optional[str] = None,
        track_name: Optional[str] = None
    ) -> pd.DataFrame:
        weather = self._sample_weather_scenario(track_type)
        has_safety_car = self._check_safety_car(track_name, track_type)

        results = []
        sampled_positions = []

        for driver in drivers:
            team = teams.get(driver, 'Unknown')

            if self._check_dnf(driver, team, track_type):
                results.append({
                    'driver': driver,
                    'team': team,
                    'sampled_position': 21,
                    'dnf': True
                })
                sampled_positions.append(21)
            else:
                ml_pred = ml_predictions.get(driver) if ml_predictions else None
                position = self._sample_position(driver, team, ml_pred, track_type)

                position = self._apply_wet_weather_adjustment(driver, position, weather['is_wet'])

                if weather['is_wet']:
                    position += self.rng.normal(0, weather['chaos_multiplier'])

                results.append({
                    'driver': driver,
                    'team': team,
                    'sampled_position': position,
                    'dnf': False
                })
                sampled_positions.append(position)

        if has_safety_car:
            non_dnf_indices = [i for i, pos in enumerate(sampled_positions) if pos != 21]
            non_dnf_positions = [sampled_positions[i] for i in non_dnf_indices]

            adjusted_positions = self._apply_safety_car_chaos(non_dnf_positions, track_type)

            for idx, result_idx in enumerate(non_dnf_indices):
                results[result_idx]['sampled_position'] = adjusted_positions[idx]

        results_df = pd.DataFrame(results)

        finishers = results_df[results_df['dnf'] == False].copy()
        dnf_drivers = results_df[results_df['dnf'] == True].copy()

        if len(finishers) > 0:
            finishers['position'] = finishers['sampled_position'].rank(method='first').astype(int)
            finishers['points'] = finishers['position'].apply(
                lambda pos: POINTS_SYSTEM.get(pos, 0)
            )

        if len(dnf_drivers) > 0:
            dnf_drivers['position'] = 21
            dnf_drivers['points'] = 0

        results_df = pd.concat([finishers, dnf_drivers], ignore_index=True)

        return results_df[['driver', 'team', 'position', 'points']]

    def simulate_championship(
        self,
        remaining_races: List[Dict],
        current_points: Dict[str, float],
        driver_teams: Dict[str, str]
    ) -> Dict:
        mode = 'HYBRID (ML + MC)' if self.use_ml_baseline else 'PURE HISTORICAL'
        logger.info(f"Monte Carlo simulation: {self.n_simulations:,} iterations, {len(remaining_races)} races ({mode})")

        drivers = list(driver_teams.keys())

        all_final_points = {driver: [] for driver in drivers}
        all_final_positions = {driver: [] for driver in drivers}

        for sim_idx in range(self.n_simulations):
            if (sim_idx + 1) % 2000 == 0:
                logger.info(f"  {sim_idx + 1:,}/{self.n_simulations:,} simulations...")

            sim_points = current_points.copy()

            for race_info in remaining_races:
                race_result = self.simulate_race(
                    drivers=drivers,
                    teams=driver_teams,
                    ml_predictions=race_info.get('ml_predictions'),
                    track_type=race_info.get('track_type'),
                    track_name=race_info.get('track_name')
                )

                for _, row in race_result.iterrows():
                    driver = row['driver']
                    sim_points[driver] = sim_points.get(driver, 0) + row['points']

            final_standings = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)

            for position, (driver, points) in enumerate(final_standings, 1):
                all_final_points[driver].append(points)
                all_final_positions[driver].append(position)

        logger.info(f"✓ Completed {self.n_simulations:,} simulations")

        results = {
            'final_points': {d: np.array(pts) for d, pts in all_final_points.items()},
            'final_positions': {d: np.array(pos) for d, pos in all_final_positions.items()}
        }

        return results

    def analyze_results(
        self,
        simulation_results: Dict,
        current_points: Dict[str, float]
    ) -> pd.DataFrame:
        final_points = simulation_results['final_points']
        final_positions = simulation_results['final_positions']

        analysis = []

        for driver in final_points.keys():
            points_array = final_points[driver]
            positions_array = final_positions[driver]

            analysis.append({
                'driver': driver,
                'current_points': current_points.get(driver, 0),
                'expected_points': points_array.mean(),
                'points_std': points_array.std(),
                'points_min': points_array.min(),
                'points_max': points_array.max(),
                'points_median': np.median(points_array),
                'expected_position': positions_array.mean(),
                'position_std': positions_array.std(),
                'median_position': np.median(positions_array),
                'prob_win': (positions_array == 1).sum() / len(positions_array),
                'prob_podium': (positions_array <= 3).sum() / len(positions_array),
                'prob_top5': (positions_array <= 5).sum() / len(positions_array),
                'prob_top10': (positions_array <= 10).sum() / len(positions_array),
            })

        df = pd.DataFrame(analysis)
        df = df.sort_values('expected_position')

        logger.info("\nExpected final standings:")
        display_df = df[['driver', 'expected_position', 'prob_win', 'prob_podium', 'expected_points']].copy()
        display_df['prob_win'] = (display_df['prob_win'] * 100).round(1)
        display_df['prob_podium'] = (display_df['prob_podium'] * 100).round(1)
        display_df['expected_position'] = display_df['expected_position'].round(2)
        display_df['expected_points'] = display_df['expected_points'].round(1)
        display_df.columns = ['Driver', 'Exp. Pos', 'Win %', 'Podium %', 'Exp. Points']
        logger.info("\n" + display_df.head(10).to_string(index=False))

        logger.info("\nChampionship win probabilities:")
        top_contenders = df.nlargest(5, 'prob_win')
        for _, row in top_contenders.iterrows():
            prob_pct = row['prob_win'] * 100
            logger.info(f"  {row['driver']}: {prob_pct:.2f}%")

        logger.info("\nTop 3 finish probabilities:")
        top_podium = df.nlargest(8, 'prob_podium')
        for _, row in top_podium.iterrows():
            prob_pct = row['prob_podium'] * 100
            logger.info(f"  {row['driver']}: {prob_pct:.2f}%")

        return df

    def generate_confidence_intervals(
        self,
        simulation_results: Dict,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        final_points = simulation_results['final_points']
        final_positions = simulation_results['final_positions']

        intervals = []

        for driver in final_points.keys():
            points_array = final_points[driver]
            positions_array = final_positions[driver]

            intervals.append({
                'driver': driver,
                'points_lower': np.percentile(points_array, lower_percentile),
                'points_upper': np.percentile(points_array, upper_percentile),
                'position_lower': np.percentile(positions_array, lower_percentile),
                'position_upper': np.percentile(positions_array, upper_percentile),
            })

        df = pd.DataFrame(intervals)

        logger.info(f"\n{confidence_level*100:.0f}% Confidence Intervals:")
        logger.info("\n" + df.head(10).to_string(index=False))

        return df


def run_monte_carlo_prediction(
    historical_data: pd.DataFrame,
    remaining_races: List[Dict],
    current_standings: pd.DataFrame,
    driver_teams: Dict[str, str],
    n_simulations: int = 10000,
    use_ml_baseline: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    current_points = {}
    if not current_standings.empty:
        current_points = dict(zip(
            current_standings['driver'],
            current_standings['points']
        ))

    for driver in driver_teams.keys():
        if driver not in current_points:
            current_points[driver] = 0

    simulator = MonteCarloSimulator(
        historical_data=historical_data,
        n_simulations=n_simulations,
        use_ml_baseline=use_ml_baseline
    )

    simulation_results = simulator.simulate_championship(
        remaining_races=remaining_races,
        current_points=current_points,
        driver_teams=driver_teams
    )

    analysis_df = simulator.analyze_results(simulation_results, current_points)

    simulator.generate_confidence_intervals(simulation_results, confidence_level=0.95)

    return analysis_df, simulation_results
