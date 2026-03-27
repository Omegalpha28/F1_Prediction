import logging
import pandas as pd

logger = logging.getLogger(__name__)
SCHEMA: dict[str, dict] = {
    "circuits": {
        "columns":  ["circuitId", "circuitRef", "name", "location", "country", "lat", "lng", "alt", "url"],
        "critical": ["circuitId", "lat", "lng"],
        "types": {"circuitId": "Int64", "lat": "Float64", "lng": "Float64", "alt": "Float64",
                  "circuitRef": "string", "name": "string", "location": "string", "country": "string"}
    },
    "constructor_results": {
        "columns":  ["constructorResultsId", "raceId", "constructorId", "points", "status"],
        "critical": ["raceId", "constructorId", "points"],
        "types": {"constructorResultsId": "Int64", "raceId": "Int64", "constructorId": "Int64",
                  "points": "Float64", "status": "string"}
    },
    "constructor_standings": {
        "columns":  ["constructorStandingsId", "raceId", "constructorId", "points", "position", "positionText", "wins"],
        "critical": ["raceId", "constructorId", "points", "position"],
        "types": {"constructorStandingsId": "Int64", "raceId": "Int64", "constructorId": "Int64",
                  "points": "Float64", "position": "Int64", "wins": "Int64", "positionText": "string"}
    },
    "constructors": {
        "columns":  ["constructorId", "constructorRef", "name", "nationality", "url"],
        "critical": ["constructorId", "name"],
        "types": {"constructorId": "Int64", "constructorRef": "string", "name": "string", "nationality": "string"}
    },
    "driver_standings": {
        "columns":  ["driverStandingsId", "raceId", "driverId", "points", "position", "positionText", "wins"],
        "critical": ["raceId", "driverId", "points", "position"],
        "types": {"driverStandingsId": "Int64", "raceId": "Int64", "driverId": "Int64",
                  "points": "Float64", "position": "Int64", "wins": "Int64", "positionText": "string"}
    },
    "drivers": {
        "columns":  ["driverId", "driverRef", "number", "code", "forename", "surname", "dob", "nationality", "url"],
        "critical": ["driverId", "forename", "surname"],
        "types": {"driverId": "Int64", "number": "Int64", "driverRef": "string", "code": "string",
                  "forename": "string", "surname": "string", "dob": "string", "nationality": "string"}
    },
    "lap_times": {
        "columns":  ["raceId", "driverId", "lap", "position", "time", "milliseconds"],
        "critical": ["raceId", "driverId", "lap", "milliseconds"],
        "types": {"raceId": "Int64", "driverId": "Int64", "lap": "Int64", "position": "Int64",
                  "milliseconds": "Int64", "time": "string"}
    },
    "pit_stops": {
        "columns":  ["raceId", "driverId", "stop", "lap", "time", "duration", "milliseconds"],
        "critical": ["raceId", "driverId", "lap", "milliseconds"],
        "types": {"raceId": "Int64", "driverId": "Int64", "stop": "Int64", "lap": "Int64",
                  "milliseconds": "Int64", "duration": "string", "time": "string"}
    },
    "qualifying": {
        "columns":  ["qualifyId", "raceId", "driverId", "constructorId", "number", "position", "q1", "q2", "q3"],
        "critical": ["raceId", "driverId", "constructorId", "position"],
        "types": {"qualifyId": "Int64", "raceId": "Int64", "driverId": "Int64", "constructorId": "Int64",
                  "number": "Int64", "position": "Int64", "q1": "string", "q2": "string", "q3": "string"}
    },
    "races": {
        "columns":  ["raceId", "year", "round", "circuitId", "name", "date", "time", "url",
                     "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time",
                     "quali_date", "quali_time", "sprint_date", "sprint_time"],
        "critical": ["raceId", "year", "round", "circuitId"],
        "types": {"raceId": "Int64", "year": "Int64", "round": "Int64", "circuitId": "Int64"}
    },
    "results": {
        "columns":  ["resultId", "raceId", "driverId", "constructorId", "number", "grid", "position",
                     "positionText", "positionOrder", "points", "laps", "time", "milliseconds",
                     "fastestLap", "rank", "fastestLapTime", "fastestLapSpeed", "statusId"],
        "critical": ["raceId", "driverId", "constructorId", "grid", "positionOrder"],
        "types": {"resultId": "Int64", "raceId": "Int64", "driverId": "Int64", "constructorId": "Int64",
                  "number": "Int64", "grid": "Int64", "position": "Int64", "positionOrder": "Int64",
                  "points": "Float64", "laps": "Int64", "milliseconds": "Int64", "fastestLap": "Int64",
                  "rank": "Int64", "fastestLapSpeed": "Float64", "statusId": "Int64", "positionText": "string"}
    },
    "seasons": {
        "columns":  ["year", "url"],
        "critical": ["year"],
        "types": {"year": "Int64", "url": "string"}
    },
    "sprint_results": {
        "columns":  ["resultId", "raceId", "driverId", "constructorId", "number", "grid", "position",
                     "positionText", "positionOrder", "points", "laps", "time", "milliseconds",
                     "fastestLap", "fastestLapTime", "statusId"],
        "critical": ["raceId", "driverId", "positionOrder"],
        "types": {"resultId": "Int64", "raceId": "Int64", "driverId": "Int64", "constructorId": "Int64",
                  "number": "Int64", "grid": "Int64", "position": "Int64", "positionOrder": "Int64",
                  "points": "Float64", "laps": "Int64", "milliseconds": "Int64", "fastestLap": "Int64",
                  "statusId": "Int64", "positionText": "string"}
    },
    "status": {
        "columns":  ["statusId", "status"],
        "critical": ["statusId", "status"],
        "types": {"statusId": "Int64", "status": "string"}
    },
}

BLOCKING_TABLES: set[str] = {"races", "results", "drivers", "constructors"}
NAN_THRESHOLD = 0.30
TableStatus  = dict[str, str]
HealthReport = dict[str, dict]

def cast_dataframe_types(key: str, df: pd.DataFrame) -> pd.DataFrame:
    df_clean   = df.replace(r"\\N", pd.NA, regex=False)
    type_rules = SCHEMA.get(key, {}).get("types", {})
    for col, data_type in type_rules.items():
        if col in df_clean.columns:
            try:
                df_clean[col] = df_clean[col].astype(data_type)
            except Exception as e:
                logger.error(f"[{key}] Impossible de convertir '{col}' en {data_type} : {e}")
    return df_clean

def _classify_single_table(key: str, df: pd.DataFrame | None) -> str:
    if df is None:
        return "absent"
    if df.empty:
        return "vide"
    for col in SCHEMA.get(key, {}).get("critical", []):
        if col not in df.columns:
            logger.warning(f"[classify] {key} — colonne critique absente : '{col}'")
            return "suspect"
        if df[col].isna().mean() > NAN_THRESHOLD:
            logger.warning(f"[classify] {key}.{col} — NaN au-dessus du seuil")
            return "suspect"
    return "ok"

def classify_tables(dataframes: dict[str, pd.DataFrame | None]) -> TableStatus:
    return {key: _classify_single_table(key, df) for key, df in dataframes.items()}

def _build_single_report(key: str, df: pd.DataFrame | None, status_val: str) -> dict:
    schema   = SCHEMA.get(key, {})
    expected = schema.get("columns", [])
    critical = schema.get("critical", [])
    if df is None:
        return {"etat": status_val, "lignes": None, "colonnes_ok": [],
                "colonnes_absentes": expected, "nan_critiques": {}, "bloquante": key in BLOCKING_TABLES}
    missing = [c for c in expected if c not in df.columns]
    ok_cols = [c for c in expected if c in df.columns]
    nan_critiques = {
        c: round(df[c].isna().mean(), 4) for c in critical
        if c in df.columns and df[c].isna().mean() > 0
    }
    return {"etat": status_val, "lignes": len(df), "colonnes_ok": ok_cols,
            "colonnes_absentes": missing, "nan_critiques": nan_critiques, "bloquante": key in BLOCKING_TABLES}

def build_health_report(dataframes: dict, status: TableStatus) -> HealthReport:
    return {key: _build_single_report(key, df, status[key]) for key, df in dataframes.items()}

def _print_report_section(state: str, entries: list, icon: str) -> None:
    if not entries: return
    for key, info in entries:
        bloc   = " [BLOQUANTE]" if info["bloquante"] else ""
        lignes = f"{info['lignes']} lignes" if info["lignes"] is not None else "—"
        if info["colonnes_absentes"]:
            print(f"      Colonnes absentes : {', '.join(info['colonnes_absentes'])}")
        if info["nan_critiques"]:
            details = ", ".join(f"{c} {r:.0%}" for c, r in info["nan_critiques"].items())
            print(f"      NaN critiques     : {details}")

def print_health_report(report: HealthReport) -> None:
    icons   = {"ok": "✓", "suspect": "⚠", "vide": "○", "absent": "✗"}
    grouped = {"ok": [], "suspect": [], "vide": [], "absent": []}
    for key, info in report.items():
        grouped[info["etat"]].append((key, info))
    for state in ["ok", "suspect", "vide", "absent"]:
        _print_report_section(state, grouped[state], icons[state])

def _check_blocking_tables(status: TableStatus) -> None:
    for key in BLOCKING_TABLES:
        if status.get(key) != "ok":
            raise RuntimeError(f"[filter] Table bloquante '{key}' inutilisable. Arrêt.")

def _clean_single_table(key: str, df: pd.DataFrame) -> pd.DataFrame:
    critical_cols = set(SCHEMA.get(key, {}).get("critical", []))
    cols_to_drop  = [
        col for col in df.columns
        if col not in critical_cols and df[col].isna().mean() > NAN_THRESHOLD
    ]
    if cols_to_drop:
        logger.info(f"[filter] {key} — colonnes retirées : {cols_to_drop}")
        return df.drop(columns=cols_to_drop)
    return df.copy()

def filter_training_data(dataframes: dict, status: TableStatus) -> dict:
    _check_blocking_tables(status)
    usable = {}
    for key, df in dataframes.items():
        if status[key] == "ok":
            usable[key] = _clean_single_table(key, df)
        else:
            logger.info(f"[filter] '{key}' exclue (état : {status[key]})")
    logger.info(f"[filter] {len(usable)}/{len(dataframes)} tables retenues.")
    return usable

def _prompt_user_on_suspects(status: TableStatus) -> None:
    suspects = [k for k, s in status.items() if s == "suspect"]
    if suspects:
        print(f"Tables suspectes détectées : {suspects}")
        answer = input("Continuer malgré les anomalies ? [o/N] : ").strip().lower()
        if answer not in ("o", "oui", "y", "yes"):
            raise SystemExit("Chargement annulé par l'utilisateur.")

def parse_and_audit(dataframes: dict[str, pd.DataFrame | None]) -> dict[str, pd.DataFrame]:
    dataframes = {key: cast_dataframe_types(key, df) for key, df in dataframes.items() if df is not None}
    status  = classify_tables(dataframes)
    report  = build_health_report(dataframes, status)
    print_health_report(report)
    _prompt_user_on_suspects(status)
    return filter_training_data(dataframes, status)