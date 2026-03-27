import pandas as pd


def get_all_drivers(data: dict) -> pd.DataFrame:
    return data.get("drivers", pd.DataFrame())

def get_all_driver_standings(data: dict) -> pd.DataFrame:
    return data.get("driver_standings", pd.DataFrame())

def get_driver_by_id(data: dict, driver_id) -> pd.DataFrame:
    return _filter(data, "drivers", driverId=driver_id)

def get_driver_by_ref(data: dict, driver_ref: str) -> pd.DataFrame:
    return _filter(data, "drivers", driverRef=driver_ref)

def get_drivers_by_nationality(data: dict, nationality: str) -> pd.DataFrame:
    return _filter(data, "drivers", nationality=nationality)

def get_drivers_by_name(data: dict, forename: str = None, surname: str = None) -> pd.DataFrame:
    df = data.get("drivers", pd.DataFrame())
    if forename: df = df[df["forename"] == forename]
    if surname:  df = df[df["surname"]  == surname]
    return df

def get_driver_standings_by_race(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "driver_standings", raceId=race_id)

def get_driver_standings_by_driver(data: dict, driver_id) -> pd.DataFrame:
    return _filter(data, "driver_standings", driverId=driver_id)

def get_driver_history_before(data: dict, driver_id: int, current_year: int):
    from getters.get_races import get_all_races
    res   = _filter(data, "results", driverId=driver_id).copy()
    if res.empty:
        return None
    races = get_all_races(data)[["raceId", "year"]].copy()
    races["year"] = pd.to_numeric(races["year"], errors="coerce")
    res   = res.merge(races, on="raceId", how="inner")
    past  = res[res["year"] < current_year].copy()
    if past.empty:
        return None
    past["positionOrder"] = pd.to_numeric(past["positionOrder"], errors="coerce")
    return past["positionOrder"].mean()

def _filter(data: dict, table: str, **filters) -> pd.DataFrame:
    df = data.get(table, pd.DataFrame())
    for key, value in filters.items():
        if key not in df.columns:
            return pd.DataFrame()
        df = df[df[key] == value]
    return df