import pandas as pd
from main import load_and_audit
from getters.training_matrix import build_training_matrix
from getters.get_circuit      import *
from getters.get_constructors  import *
from getters.get_pilots        import *
from getters.get_races         import *
from getters.get_strategy      import *
from getters.get_weather       import *

class F1API:
    def __init__(self):
        self.data = load_and_audit()

    def get_all_drivers(self):
        return get_all_drivers(self.data)

    def get_all_circuits(self):
        return get_all_circuits(self.data)

    def get_all_constructors(self):
        return get_all_constructors(self.data)

    def get_all_constructor_results(self):
        return get_all_constructor_results(self.data)

    def get_all_constructor_standings(self):
        return get_all_constructor_standings(self.data)

    def get_all_driver_standings(self):
        return get_all_driver_standings(self.data)

    def get_all_lap_times(self):
        return get_all_lap_times(self.data)

    def get_all_pit_stops(self):
        return get_all_pit_stops(self.data)

    def get_all_qualifying(self):
        return get_all_qualifying(self.data)

    def get_all_races(self):
        return get_all_races(self.data)

    def get_all_results(self):
        return get_all_results(self.data)

    def get_all_seasons(self):
        return get_all_seasons(self.data)

    def get_all_sprint_results(self):
        return get_all_sprint_results(self.data)

    def get_all_status(self):
        return get_all_status(self.data)

    def get_driver_by_id(self, v):
        return get_driver_by_id(self.data, v)

    def get_driver_by_ref(self, v):
        return get_driver_by_ref(self.data, v)

    def get_drivers_by_nationality(self, v):
        return get_drivers_by_nationality(self.data, v)

    def get_drivers_by_name(self, f=None, s=None):
        return get_drivers_by_name(self.data, f, s)

    def get_circuit_by_id(self, v):
        return get_circuit_by_id(self.data, v)

    def get_circuit_by_ref(self, v):
        return get_circuit_by_ref(self.data, v)

    def get_circuits_by_country(self, v):
        return get_circuits_by_country(self.data, v)

    def get_constructor_by_id(self, v):
        return get_constructor_by_id(self.data, v)

    def get_constructor_by_ref(self, v):
        return get_constructor_by_ref(self.data, v)

    def get_constructors_by_nationality(self, v):
        return get_constructors_by_nationality(self.data, v)

    def get_constructor_results_by_race(self, v):
        return get_constructor_results_by_race(self.data, v)

    def get_constructor_results_by_constructor(self, v):
        return get_constructor_results_by_constructor(self.data, v)

    def get_constructor_standings_by_race(self, v):
        return get_constructor_standings_by_race(self.data, v)

    def get_constructor_standings_by_constructor(self, v):
        return get_constructor_standings_by_constructor(self.data, v)

    def get_race_by_id(self, v):
        return get_race_by_id(self.data, v)

    def get_races_by_year(self, v):
        return get_races_by_year(self.data, v)

    def get_races_by_circuit(self, v):
        return get_races_by_circuit(self.data, v)

    def get_races_by_name(self, v):
        return get_races_by_name(self.data, v) if hasattr(self.data.get("races", pd.DataFrame()), "name") else pd.DataFrame()

    def get_result_by_id(self, v):
        return get_result_by_id(self.data, v)

    def get_results_by_race(self, v):
        return get_results_by_race(self.data, v)

    def get_results_by_driver(self, v):
        return get_results_by_driver(self.data, v)

    def get_results_by_constructor(self, v):
        return get_results_by_constructor(self.data, v)

    def get_results_by_status(self, v):
        return get_results_by_status(self.data, v)

    def get_sprint_results_by_race(self, v):
        return get_sprint_results_by_race(self.data, v)

    def get_sprint_results_by_driver(self, v):
        return get_sprint_results_by_driver(self.data, v)

    def get_driver_standings_by_race(self, v):
        return get_driver_standings_by_race(self.data, v)

    def get_driver_standings_by_driver(self, v):
        return get_driver_standings_by_driver(self.data, v)

    def get_pit_stops_by_race(self, v):
        return get_pit_stops_by_race(self.data, v)

    def get_pit_stops_by_driver(self, v):
        return get_pit_stops_by_driver(self.data, v)

    def get_qualifying_by_race(self, v):
        return get_qualifying_by_race(self.data, v)

    def get_qualifying_by_driver(self, v):
        return get_qualifying_by_driver(self.data, v)

    def get_qualifying_by_constructor(self, v):
        return get_qualifying_by_constructor(self.data, v)

    def get_season_by_year(self, v):
        return get_season_by_year(self.data, v)

    def get_lap_times_by_race(self, v):
        return get_lap_times_by_race(self.data, v)

    def get_lap_times_by_driver(self, v):
        return get_lap_times_by_driver(self.data, v)

    def get_driver_history_before(self, driver_id, year):
        return get_driver_history_before(self.data, driver_id, year)

    def get_constructor_history_before(self, constructor_id, year):
        return get_constructor_history_before(self.data, constructor_id, year)

    def get_training_matrix(self) -> pd.DataFrame:
        return build_training_matrix(self)

    def get_table(self, table_name: str) -> pd.DataFrame:
        return self.data.get(table_name, pd.DataFrame())

    def get_table_by(self, table_name: str, **filters) -> pd.DataFrame:
        df = self.data.get(table_name, pd.DataFrame())
        if df.empty:
            return df
        for key, value in filters.items():
            if key not in df.columns:
                return pd.DataFrame()
            df = df[df[key] == value]
        return df