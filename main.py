# main.py
import os
import pandas as pd

def load_data() -> dict:
    base_dir = os.path.dirname(__file__)
    files_map = {
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
        "status": "status.csv"
    }

    dataframes = {}
    for key, filename in files_map.items():
        path = os.path.join(base_dir, "stats", filename)
        if os.path.exists(path):
            dataframes[key] = pd.read_csv(path)
        else:
            print(f"Warning: File not found -> {path}")
            dataframes[key] = pd.DataFrame()
    return dataframes

if __name__ == "__main__":
    dfs = load_data()
    print("Database loaded with success. Keys available:", list(dfs.keys()))
    import front
    # front.run_dashboard() is broken anyway because front.py has run_f1_explorer(), but let's keep it functionally equivalent
    front.run_f1_explorer()