import pandas as pd


def get_all_circuits(data: dict) -> pd.DataFrame:
    return data.get("circuits", pd.DataFrame())

def get_circuit_by_id(data: dict, circuit_id) -> pd.DataFrame:
    return _filter(data, "circuits", circuitId=circuit_id)

def get_circuit_by_ref(data: dict, circuit_ref: str) -> pd.DataFrame:
    return _filter(data, "circuits", circuitRef=circuit_ref)

def get_circuits_by_country(data: dict, country: str) -> pd.DataFrame:
    return _filter(data, "circuits", country=country)

def _filter(data: dict, table: str, **filters) -> pd.DataFrame:
    df = data.get(table, pd.DataFrame())
    for key, value in filters.items():
        if key not in df.columns:
            return pd.DataFrame()
        df = df[df[key] == value]
    return df