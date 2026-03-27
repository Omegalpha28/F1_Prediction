import pandas as pd


def get_all_races(data: dict) -> pd.DataFrame:
    return data.get("races", pd.DataFrame())

def get_all_results(data: dict) -> pd.DataFrame:
    return data.get("results", pd.DataFrame())

def get_all_sprint_results(data: dict) -> pd.DataFrame:
    return data.get("sprint_results", pd.DataFrame())

def get_all_status(data: dict) -> pd.DataFrame:
    return data.get("status", pd.DataFrame())

def get_races_by_name(data: dict, name: str) -> pd.DataFrame:
    return _filter(data, "races", name=name)

def get_race_by_id(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "races", raceId=race_id)

def get_races_by_year(data: dict, year) -> pd.DataFrame:
    return _filter(data, "races", year=year)

def get_races_by_circuit(data: dict, circuit_id) -> pd.DataFrame:
    return _filter(data, "races", circuitId=circuit_id)

def get_result_by_id(data: dict, result_id) -> pd.DataFrame:
    return _filter(data, "results", resultId=result_id)

def get_results_by_race(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "results", raceId=race_id)

def get_results_by_driver(data: dict, driver_id) -> pd.DataFrame:
    return _filter(data, "results", driverId=driver_id)

def get_results_by_constructor(data: dict, constructor_id) -> pd.DataFrame:
    return _filter(data, "results", constructorId=constructor_id)

def get_results_by_status(data: dict, status_id) -> pd.DataFrame:
    return _filter(data, "results", statusId=status_id)

def get_sprint_results_by_race(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "sprint_results", raceId=race_id)

def get_sprint_results_by_driver(data: dict, driver_id) -> pd.DataFrame:
    return _filter(data, "sprint_results", driverId=driver_id)

def _filter(data: dict, table: str, **filters) -> pd.DataFrame:
    df = data.get(table, pd.DataFrame())
    for key, value in filters.items():
        if key not in df.columns:
            return pd.DataFrame()
        df = df[df[key] == value]
    return df