import pandas as pd


def get_all_seasons(data: dict) -> pd.DataFrame:
    return data.get("seasons", pd.DataFrame())

def get_all_lap_times(data: dict) -> pd.DataFrame:
    return data.get("lap_times", pd.DataFrame())

def get_season_by_year(data: dict, year) -> pd.DataFrame:
    return _filter(data, "seasons", year=year)

def get_lap_times_by_race(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "lap_times", raceId=race_id)

def get_lap_times_by_driver(data: dict, driver_id) -> pd.DataFrame:
    return _filter(data, "lap_times", driverId=driver_id)

def _filter(data: dict, table: str, **filters) -> pd.DataFrame:
    df = data.get(table, pd.DataFrame())
    for key, value in filters.items():
        if key not in df.columns:
            return pd.DataFrame()
        df = df[df[key] == value]
    return df