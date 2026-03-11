# api.py
import pandas as pd
from main import load_data
from prediction import F1Predictor

class F1API:
    def __init__(self):
        self.data = load_data()
        self.predictor = F1Predictor()

    # --- GETTERS ---
    def get_driver_info(self, driver_ref: str):
        drivers_df = self.data.get("drivers", pd.DataFrame())
        if not drivers_df.empty and 'driverRef' in drivers_df.columns:
            return drivers_df[drivers_df['driverRef'] == driver_ref]
        return pd.DataFrame()

    def get_all_drivers(self):
        return self.data.get("drivers", pd.DataFrame())

    def get_all_circuits(self):
        return self.data.get("circuits", pd.DataFrame())

    def get_all_constructor_results(self):
        return self.data.get("constructor_results", pd.DataFrame())

    def get_all_constructor_standings(self):
        return self.data.get("constructor_standings", pd.DataFrame())

    def get_all_driver_standings(self):
        return self.data.get("driver_standings", pd.DataFrame())

    def get_all_lap_times(self):
        return self.data.get("lap_times", pd.DataFrame())

    def get_all_pit_stops(self):
        return self.data.get("pit_stops", pd.DataFrame())

    def get_all_qualifying(self):
        return self.data.get("qualifying", pd.DataFrame())

    def get_all_races(self):
        return self.data.get("races", pd.DataFrame())

    def get_all_results(self):
        return self.data.get("results", pd.DataFrame())

    def get_all_seasons(self):
        return self.data.get("seasons", pd.DataFrame())

    def get_all_sprint_results(self):
        return self.data.get("sprint_results", pd.DataFrame())

    def get_all_status(self):
        return self.data.get("status", pd.DataFrame())

    def get_table(self, table_name: str):
        return self.data.get(table_name, pd.DataFrame())

    def get_table_by(self, table_name: str, **filters):
        df = self.data.get(table_name, pd.DataFrame())
        if df.empty:
            return df
        for key, value in filters.items():
            if key not in df.columns:
                return pd.DataFrame()
            df = df[df[key] == value]
        return df

    # --- GETTERS VARIANTS DRIVERS ---
    def get_driver_by_id(self, driver_id):
        return self.get_table_by("drivers", driverId=driver_id)

    def get_driver_by_ref(self, driver_ref: str):
        return self.get_table_by("drivers", driverRef=driver_ref)

    def get_drivers_by_nationality(self, nationality: str):
        return self.get_table_by("drivers", nationality=nationality)

    def get_drivers_by_name(self, forename: str = None, surname: str = None):
        df = self.get_all_drivers()
        if df.empty:
            return df
        if forename:
            df = df[df["forename"] == forename]
        if surname:
            df = df[df["surname"] == surname]
        return df

    # --- GETTERS VARIANTS CIRCUITS ---
    def get_circuit_by_id(self, circuit_id):
        return self.get_table_by("circuits", circuitId=circuit_id)

    def get_circuit_by_ref(self, circuit_ref: str):
        return self.get_table_by("circuits", circuitRef=circuit_ref)

    def get_circuits_by_country(self, country: str):
        return self.get_table_by("circuits", country=country)

    # --- GETTERS VARIANTS CONSTRUCTORS ---
    def get_constructor_by_id(self, constructor_id):
        return self.get_table_by("constructors", constructorId=constructor_id)

    def get_constructor_by_ref(self, constructor_ref: str):
        return self.get_table_by("constructors", constructorRef=constructor_ref)

    def get_constructors_by_nationality(self, nationality: str):
        return self.get_table_by("constructors", nationality=nationality)

    # --- GETTERS VARIANTS RACES ---
    def get_race_by_id(self, race_id):
        return self.get_table_by("races", raceId=race_id)

    def get_races_by_year(self, year):
        return self.get_table_by("races", year=year)

    def get_races_by_circuit(self, circuit_id):
        return self.get_table_by("races", circuitId=circuit_id)

    def get_races_by_name(self, name: str):
        return self.get_table_by("races", name=name)

    # --- GETTERS VARIANTS RESULTS ---
    def get_result_by_id(self, result_id):
        return self.get_table_by("results", resultId=result_id)

    def get_results_by_race(self, race_id):
        return self.get_table_by("results", raceId=race_id)

    def get_results_by_driver(self, driver_id):
        return self.get_table_by("results", driverId=driver_id)

    def get_results_by_constructor(self, constructor_id):
        return self.get_table_by("results", constructorId=constructor_id)

    def get_results_by_status(self, status_id):
        return self.get_table_by("results", statusId=status_id)

    # --- GETTERS VARIANTS SPRINT RESULTS ---
    def get_sprint_results_by_race(self, race_id):
        return self.get_table_by("sprint_results", raceId=race_id)

    def get_sprint_results_by_driver(self, driver_id):
        return self.get_table_by("sprint_results", driverId=driver_id)

    # --- GETTERS VARIANTS CONSTRUCTOR RESULTS ---
    def get_constructor_results_by_race(self, race_id):
        return self.get_table_by("constructor_results", raceId=race_id)

    def get_constructor_results_by_constructor(self, constructor_id):
        return self.get_table_by("constructor_results", constructorId=constructor_id)

    # --- GETTERS VARIANTS STANDINGS ---
    def get_driver_standings_by_race(self, race_id):
        return self.get_table_by("driver_standings", raceId=race_id)

    def get_driver_standings_by_driver(self, driver_id):
        return self.get_table_by("driver_standings", driverId=driver_id)

    def get_constructor_standings_by_race(self, race_id):
        return self.get_table_by("constructor_standings", raceId=race_id)

    def get_constructor_standings_by_constructor(self, constructor_id):
        return self.get_table_by("constructor_standings", constructorId=constructor_id)

    # --- GETTERS VARIANTS LAP & PIT & QUALIFICATION ---
    def get_lap_times_by_race(self, race_id):
        return self.get_table_by("lap_times", raceId=race_id)

    def get_lap_times_by_driver(self, driver_id):
        return self.get_table_by("lap_times", driverId=driver_id)

    def get_pit_stops_by_race(self, race_id):
        return self.get_table_by("pit_stops", raceId=race_id)

    def get_pit_stops_by_driver(self, driver_id):
        return self.get_table_by("pit_stops", driverId=driver_id)

    def get_qualifying_by_race(self, race_id):
        return self.get_table_by("qualifying", raceId=race_id)

    def get_qualifying_by_driver(self, driver_id):
        return self.get_table_by("qualifying", driverId=driver_id)

    def get_qualifying_by_constructor(self, constructor_id):
        return self.get_table_by("qualifying", constructorId=constructor_id)

    # --- GETTERS VARIANTS SEASONS & STATUS ---
    def get_season_by_year(self, year):
        return self.get_table_by("seasons", year=year)

    def get_status_by_id(self, status_id):
        return self.get_table_by("status", statusId=status_id)

    # --- LOGIQUE ML ---
    def setup_prediction_model(self):
        results_df = self.data.get("results", pd.DataFrame())
        self.predictor.train(results_df)

    def simulate_race_result(self, starting_grid_position: int) -> float:
        return self.predictor.predict_position(starting_grid_position)