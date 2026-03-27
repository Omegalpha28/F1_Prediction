import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

FILES_MAP = {
    "circuits":              "circuits.csv",
    "constructors":          "constructors.csv",
    "constructor_results":   "constructor_results.csv",
    "constructor_standings": "constructor_standings.csv",
    "driver_standings":      "driver_standings.csv",
    "drivers":               "drivers.csv",
    "lap_times":             "lap_times.csv",
    "pit_stops":             "pit_stops.csv",
    "qualifying":            "qualifying.csv",
    "races":                 "races.csv",
    "results":               "results.csv",
    "seasons":               "seasons.csv",
    "sprint_results":        "sprint_results.csv",
    "status":                "status.csv",
}

def _load_single_file(key: str, path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        logger.warning(f"[load_data] Fichier introuvable : {path}")
        return None
    try:
        df = pd.read_csv(path, low_memory=False, na_values=[r"\N", "\\N"])
        logger.debug(f"[load_data] OK — {key} ({len(df)} lignes)")
        return df
    except Exception as exc:
        logger.error(f"[load_data] Échec lecture {path} : {exc}")
        return None

def load_data() -> dict[str, pd.DataFrame | None]:
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    stats_dir = os.path.abspath(os.path.join(base_dir, "..", "stats"))
    dataframes = {
        key: _load_single_file(key, os.path.join(stats_dir, filename))
        for key, filename in FILES_MAP.items()
    }
    loaded_count = sum(v is not None for v in dataframes.values())
    logger.info(f"[load_data] {loaded_count}/{len(dataframes)} tables chargées.")
    return dataframes