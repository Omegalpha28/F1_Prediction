import pandas as pd


def build_training_matrix(api) -> pd.DataFrame:
    res = _build_base(api)
    if res.empty:
        return pd.DataFrame()
    res = _merge_races(res, api)
    res = _merge_driver_standings(res, api)
    res = _merge_constructor_standings(res, api)
    res = _apply_defaults(res)
    return res


def _build_base(api) -> pd.DataFrame:
    res = api.get_all_results().copy()
    if res.empty:
        return pd.DataFrame()
    res["positionOrder"]  = pd.to_numeric(res["positionOrder"],  errors="coerce").fillna(20)
    res["grid"]           = pd.to_numeric(res["grid"],           errors="coerce").fillna(10)
    res["raceId"]         = pd.to_numeric(res["raceId"],         errors="coerce")
    res["driverId"]       = pd.to_numeric(res["driverId"],       errors="coerce")
    res["constructorId"]  = pd.to_numeric(res["constructorId"],  errors="coerce")
    return res


def _merge_races(res: pd.DataFrame, api) -> pd.DataFrame:
    races          = api.get_all_races()[["raceId", "year", "round", "circuitId"]].copy()
    races["year"]  = pd.to_numeric(races["year"],  errors="coerce")
    races["round"] = pd.to_numeric(races["round"], errors="coerce")
    return res.merge(races, on="raceId", how="left")


def _merge_driver_standings(res: pd.DataFrame, api) -> pd.DataFrame:
    drv_std = api.get_all_driver_standings().copy()
    if drv_std.empty:
        res["driver_points"] = 0.0
        return res
    drv_std["points"] = pd.to_numeric(drv_std["points"], errors="coerce").fillna(0)
    drv_std = drv_std[["raceId", "driverId", "points"]].rename(columns={"points": "driver_points"})
    res = res.merge(drv_std, on=["raceId", "driverId"], how="left")
    res["driver_points"] = res["driver_points"].fillna(0.0)
    return res


def _merge_constructor_standings(res: pd.DataFrame, api) -> pd.DataFrame:
    con_std = api.get_all_constructor_standings().copy()
    if con_std.empty:
        return res
    con_std["points"]   = pd.to_numeric(con_std["points"],   errors="coerce").fillna(0)
    con_std["wins"]     = pd.to_numeric(con_std["wins"],     errors="coerce").fillna(0)
    con_std["position"] = pd.to_numeric(con_std["position"], errors="coerce").fillna(10)
    con_std = con_std[["raceId", "constructorId", "points", "wins", "position"]].rename(columns={
        "points":   "constructor_points",
        "wins":     "constructor_wins",
        "position": "constructor_champ_pos",
    })
    return res.merge(con_std, on=["raceId", "constructorId"], how="left")


def _apply_defaults(res: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "constructor_points":   0.0,
        "constructor_wins":     0,
        "constructor_champ_pos": 10,
        "year":                 2000,
        "round":                1,
        "circuitId":            0,
    }
    for col, val in defaults.items():
        if col not in res.columns:
            res[col] = val
        else:
            res[col] = res[col].fillna(val)
    return res