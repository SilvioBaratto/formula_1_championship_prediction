#!/usr/bin/env python3
"""
Discover unique circuits from F1 seasons for track classification.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import fastf1
from src.utils.config import YEARS, CACHE_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def discover_circuits():
    all_circuits = {}

    for year in YEARS:
        try:
            schedule = fastf1.get_event_schedule(year)
            schedule = schedule[schedule['EventFormat'] != 'testing']

            for _, event in schedule.iterrows():
                location = event['Location']

                if location not in all_circuits:
                    all_circuits[location] = {
                        'location': location,
                        'country': event['Country'],
                        'event_names': [],
                        'years': []
                    }

                if event['EventName'] not in all_circuits[location]['event_names']:
                    all_circuits[location]['event_names'].append(event['EventName'])

                if year not in all_circuits[location]['years']:
                    all_circuits[location]['years'].append(year)

        except Exception as e:
            logger.error(f"Failed to fetch {year} schedule: {e}")
            continue

    logger.info(f"Found {len(all_circuits)} unique circuits")
    return all_circuits


def generate_template(circuits):
    output_file = project_root / "data" / "circuit_classification_template.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sorted_circuits = sorted(circuits.items())

    template = """
Track Types:
  high_speed_low_downforce - Long straights (Monza-style)
  street_circuit_high_downforce - Street circuits (Monaco-style)
  high_downforce_technical - Technical permanent circuits (Suzuka-style)
  medium_downforce_balanced - Mixed characteristics (Silverstone-style)
  high_altitude_special - High altitude (Mexico City)
  mixed_characteristics - Doesn't fit other categories

TRACK_CLUSTERS = {
    'high_speed_low_downforce': [],
    'street_circuit_high_downforce': [],
    'high_downforce_technical': [],
    'medium_downforce_balanced': [],
    'high_altitude_special': [],
    'mixed_characteristics': []
}

Circuits to classify:
"""

    with open(output_file, 'w') as f:
        f.write(f"Total circuits: {len(circuits)}\n\n")

        for location, info in sorted_circuits:
            f.write(f"\nLocation: {location}\n")
            f.write(f"  Country: {info['country']}\n")
            f.write(f"  Events: {', '.join(info['event_names'])}\n")
            f.write(f"  Years: {info['years']}\n")

        f.write("\n" + "="*70 + "\n")
        f.write(template)

        for location, info in sorted_circuits:
            f.write(f"# '{location}',  # {info['country']}\n")

    logger.info(f"Template saved to {output_file}")


def main():
    try:
        circuits = discover_circuits()

        if not circuits:
            logger.error("No circuits found")
            return 1

        generate_template(circuits)
        return 0

    except Exception as e:
        logger.error(f"Circuit discovery failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
