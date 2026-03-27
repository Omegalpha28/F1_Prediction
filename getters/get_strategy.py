import pandas as pd


def get_all_pit_stops(data: dict) -> pd.DataFrame:
    return data.get("pit_stops", pd.DataFrame())

def get_all_qualifying(data: dict) -> pd.DataFrame:
    return data.get("qualifying", pd.DataFrame())

def get_pit_stops_by_race(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "pit_stops", raceId=race_id)

def get_pit_stops_by_driver(data: dict, driver_id) -> pd.DataFrame:
    return _filter(data, "pit_stops", driverId=driver_id)

def get_qualifying_by_race(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "qualifying", raceId=race_id)

def get_qualifying_by_driver(data: dict, driver_id) -> pd.DataFrame:
    return _filter(data, "qualifying", driverId=driver_id)

def get_qualifying_by_constructor(data: dict, constructor_id) -> pd.DataFrame:
    return _filter(data, "qualifying", constructorId=constructor_id)

def _filter(data: dict, table: str, **filters) -> pd.DataFrame:
    df = data.get(table, pd.DataFrame())
    for key, value in filters.items():
        if key not in df.columns:
            return pd.DataFrame()
        df = df[df[key] == value]
    return df