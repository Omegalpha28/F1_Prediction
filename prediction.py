from models.Circuits import CircuitModel
from models.Races import RaceModel
from models.Strategy import StrategyModel
from models.Pilots import PilotModel
from models.WeathersConditions import WeatherModel
from models.Constructor import ConstructorModel
from api import F1API
import pandas as pd
import numpy as np

class F1Predictor:
    def __init__(self, F1API_instance: F1API):
        self.api = F1API_instance
        self.circuit_model = CircuitModel()
        self.race_model = RaceModel()
        self.strategy_model = StrategyModel()
        self.pilot_model = PilotModel()
        self.weather_model = WeatherModel()
        self.constructor_model = ConstructorModel()

    def train_all(self):
        print("--- Début de l'entraînement des modèles ---")
        circuits = self.api.get_all_circuits()
        races = self.api.get_all_races()
        self.circuit_model.train(circuits)
        self.race_model.train(races)
        self.strategy_model.train(self.api.get_all_pit_stops())
        training_matrix = self.api.get_training_matrix()
        self.pilot_model.train(training_matrix)
        self.constructor_model.train(training_matrix)
        self.weather_model.train()
        print("--- Tous les modèles sont prêts ---")

    def _get_current_constructor(self, driver_id: int, year: int) -> int:
        races_this_year = self.api.get_races_by_year(year)
        if not races_this_year.empty:
            last_race_id = races_this_year.sort_values('round', ascending=False).iloc[0]['raceId']
            quali = self.api.get_qualifying_by_race(last_race_id)
            if not quali.empty:
                driver_quali = quali[quali['driverId'] == driver_id]
                if not driver_quali.empty:
                    return int(driver_quali.iloc[0]['constructorId'])
        return 0

    def simulate_race(self, driver_id: int, circuit_id: int, grid_pos: int, weather_conditions: dict, current_year: int = 2024) -> dict:
        constructor_id = self._get_current_constructor(driver_id, current_year)
        con_std = self.api.get_constructor_standings_by_race(1110)
        con_pts, con_wins, con_pos = 0, 0, 10
        if not con_std.empty:
            row = con_std[con_std['constructorId'] == constructor_id]
            if not row.empty:
                con_pts = float(row.iloc[0]['points'])
                con_wins = int(row.iloc[0]['wins'])
                con_pos = int(row.iloc[0]['position'])

        constructor_data = {
            'constructor_points': con_pts,
            'constructor_wins': con_wins,
            'constructor_champ_pos': con_pos,
            'reliability_score': 0.95
        }

        ml_con_potential = self.constructor_model.predict(constructor_data)
        avg_pilot_pos = self.api.get_driver_history_before(driver_id, current_year)
        is_rookie = avg_pilot_pos is None
        drv_std = self.api.get_driver_standings_by_race(1110)
        driver_pts = 0.0
        if not drv_std.empty:
            d_row = drv_std[drv_std['driverId'] == driver_id]
            if not d_row.empty: driver_pts = float(d_row.iloc[0]['points'])
        pilot_data = {
            'grid': grid_pos,
            'driver_points': driver_pts,
            'constructor_points': con_pts,
            'consistency': 0.8,
        }

        ml_pilot_pred = self.pilot_model.predict(pilot_data)

        reliability = constructor_data.get('reliability_score', 0.95)
        consistency = pilot_data.get('consistency', 0.8)
        avg_con_pos = self.api.get_constructor_history_before(constructor_id, current_year)
        hist_con_score = avg_con_pos if avg_con_pos else 10.0

        if is_rookie:
            base_score = (ml_con_potential * 0.5 * reliability) + (hist_con_score * 0.3) + (ml_pilot_pred * 0.2)
        else:
            base_score = (ml_pilot_pred * 0.5 * (1 + (consistency - 0.8))) + (ml_con_potential * 0.3) + (hist_con_score * 0.2)
        overtaking_factor = self.circuit_model.get_overtaking_factor(circuit_id)
        weather_impact = self.weather_model.predict(weather_conditions)

        if weather_impact > 1.05:
            overtaking_factor = overtaking_factor / weather_impact
            uncertainty = np.random.uniform(-2, 2) * (weather_impact - 1) * 5
            base_score += uncertainty

        final_score = (grid_pos * overtaking_factor) + (base_score * (1 - overtaking_factor))
        final_position = np.clip(final_score, 1.0, 20.0)

        return {
            "driver_id": driver_id,
            "is_rookie": is_rookie,
            "expected_position_value": round(float(final_position), 2),
            "expected_position_str": f"P{final_position:.1f}",
            "factors": {
                "car_potential": round(ml_con_potential, 2),
                "pilot_performance": round(ml_pilot_pred, 2),
                "overtaking_difficulty": round(overtaking_factor, 2),
                "reliability_applied": reliability,
                "consistency_applied": consistency
            }
        }