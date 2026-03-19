import logging
import os
import pandas as pd
import subprocess
import sys

logger = logging.getLogger(__name__)

FILES_MAP = {
    "circuits": "circuits.csv",
    "constructors": "constructors.csv",
    "constructor_results": "constructor_results.csv",
    "constructor_standings": "constructor_standings.csv",
    "driver_standings": "driver_standings.csv",
    "drivers": "drivers.csv",
    "lap_times": "lap_times.csv",
    "pit_stops": "pit_stops.csv",
    "qualifying": "qualifying.csv",
    "races": "races.csv",
    "results": "results.csv",
    "seasons": "seasons.csv",
    "sprint_results": "sprint_results.csv",
    "status": "status.csv",
}


def load_data() -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stats_dir = os.path.join(base_dir, "stats")
    dataframes = {}
    for key, filename in FILES_MAP.items():
        path = os.path.join(stats_dir, filename)
        if os.path.exists(path):
            try:
                dataframes[key] = pd.read_csv(path, low_memory=False)
            except Exception as e:
                logger.error(f"Erreur lecture {path} : {e}")
                dataframes[key] = pd.DataFrame()
        else:
            logger.warning(f"Fichier introuvable : {path}")
            dataframes[key] = pd.DataFrame()
    logger.info(f"Dataset chargé — {len(dataframes)} tables disponibles.")
    return dataframes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dfs = load_data()
    loaded = [k for k, v in dfs.items() if not v.empty]
    empty = [k for k, v in dfs.items() if v.empty]

    print(f"Dataset chargé — {len(loaded)} tables OK, {len(empty)} vides.")
    if empty:
        print(f"Tables vides : {empty}")
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    print(f"\nLancement du dashboard : {dashboard_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])