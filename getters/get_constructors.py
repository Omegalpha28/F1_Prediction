import pandas as pd


def get_all_constructors(data: dict) -> pd.DataFrame:
    return data.get("constructors", pd.DataFrame())

def get_all_constructor_results(data: dict) -> pd.DataFrame:
    return data.get("constructor_results", pd.DataFrame())

def get_all_constructor_standings(data: dict) -> pd.DataFrame:
    return data.get("constructor_standings", pd.DataFrame())

def get_constructor_by_id(data: dict, constructor_id) -> pd.DataFrame:
    return _filter(data, "constructors", constructorId=constructor_id)

def get_constructor_by_ref(data: dict, constructor_ref: str) -> pd.DataFrame:
    return _filter(data, "constructors", constructorRef=constructor_ref)

def get_constructors_by_nationality(data: dict, nationality: str) -> pd.DataFrame:
    return _filter(data, "constructors", nationality=nationality)

def get_constructor_results_by_race(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "constructor_results", raceId=race_id)

def get_constructor_results_by_constructor(data: dict, constructor_id) -> pd.DataFrame:
    return _filter(data, "constructor_results", constructorId=constructor_id)

def get_constructor_standings_by_race(data: dict, race_id) -> pd.DataFrame:
    return _filter(data, "constructor_standings", raceId=race_id)

def get_constructor_standings_by_constructor(data: dict, constructor_id) -> pd.DataFrame:
    return _filter(data, "constructor_standings", constructorId=constructor_id)

def get_constructor_history_before(data: dict, constructor_id: int, current_year: int):
    from getters.get_races import get_all_races
    res   = get_all_constructors(data)
    races = get_all_races(data)[["raceId", "year"]].copy()
    races["year"] = pd.to_numeric(races["year"], errors="coerce")
    res   = _filter(data, "results", constructorId=constructor_id).copy()
    if res.empty:
        return None
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