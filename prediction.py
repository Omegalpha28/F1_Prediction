import logging
import numpy as np
import pandas as pd
from api import F1API
from models.Circuits import CircuitModel
from models.Constructor import ConstructorModel
from models.Pilots import PilotModel
from models.Races import RaceModel
from models.Strategy import StrategyModel
from models.WeathersConditions import WeatherModel

logger = logging.getLogger(__name__)


class F1Predictor:

    def __init__(self, api: F1API):
        self.api = api
        self.circuit_model = CircuitModel()
        self.race_model = RaceModel()
        self.strategy_model = StrategyModel()
        self.pilot_model = PilotModel()
        self.weather_model = WeatherModel()
        self.constructor_model = ConstructorModel()

    def train_all(self) -> None:
        logger.info("--- Début de l'entraînement des modèles ---")
        self._train_model("CircuitModel",     self.circuit_model.train,     self.api)
        self._train_model("WeatherModel",     self.weather_model.train,     self.api)
        self._train_model("ConstructorModel", self.constructor_model.train, self.api)
        self._train_model("PilotModel",       self.pilot_model.train,       self.api)
        self._train_model(
            "StrategyModel",
            self.strategy_model.train,
            self.api,
            self.circuit_model,
        )
        self._train_model(
            "RaceModel",
            self.race_model.train,
            self.api,
            self.circuit_model,
            self.constructor_model,
            self.pilot_model,
            self.weather_model,
        )

        trained = [
            name for name, model in self._all_models()
            if model.is_trained
        ]
        failed = [
            name for name, model in self._all_models()
            if not model.is_trained
        ]

        logger.info(f"Modèles entraînés : {trained}")
        if failed:
            logger.warning(f"Modèles en échec : {failed}")
        logger.info("--- Entraînement terminé ---")

    def _train_model(self, name: str, train_fn, *args) -> None:
        try:
            train_fn(*args)
            logger.info(f"[{name}] ✓ Entraîné.")
        except Exception as e:
            logger.error(f"[{name}] ✗ Échec : {e}")

    def _all_models(self) -> list[tuple[str, object]]:
        return [
            ("CircuitModel",     self.circuit_model),
            ("WeatherModel",     self.weather_model),
            ("ConstructorModel", self.constructor_model),
            ("PilotModel",       self.pilot_model),
            ("StrategyModel",    self.strategy_model),
            ("RaceModel",        self.race_model),
        ]
    def simulate_race(
        self,
        driver_id: int,
        circuit_id: int,
        grid_pos: int,
        weather_conditions: dict,
        current_year: int = 2024,
    ) -> dict:
        constructor_id = self._get_current_constructor(driver_id, current_year)
        con_pts, con_wins, con_pos = self._get_constructor_standings(
            constructor_id, current_year
        )

        constructor_data = {
            "constructor_points": con_pts,
            "constructor_wins": con_wins,
            "constructor_champ_pos": con_pos,
            "reliability_score": self._get_reliability_score(constructor_id),
        }
        ml_con_potential = self.constructor_model.predict(constructor_data)
        avg_pilot_pos = self.api.get_driver_history_before(driver_id, current_year)
        is_rookie = avg_pilot_pos is None
        driver_pts = self._get_driver_points(driver_id, current_year)
        behavioral = (
            self.pilot_model.get_driver_behavioral_features(driver_id)
            if self.pilot_model.is_trained
            else {"consistency": 0.75, "aggression": 0.10}
        )
        consistency = behavioral["consistency"]
        aggression = behavioral["aggression"]

        pilot_data = {
            "grid": grid_pos,
            "driver_points": driver_pts,
            "constructor_points": con_pts,
            "consistency": consistency,
            "aggression": aggression,
        }
        ml_pilot_pred = self.pilot_model.predict(pilot_data)
        avg_con_pos = self.api.get_constructor_history_before(
            constructor_id, current_year
        )
        hist_con_score = avg_con_pos if avg_con_pos else 10.0
        reliability = constructor_data["reliability_score"]
        if is_rookie:
            base_score = (
                (ml_con_potential * 0.5 * reliability)
                + (hist_con_score * 0.3)
                + (ml_pilot_pred * 0.2)
            )
        else:
            base_score = (
                (ml_pilot_pred * 0.5 * (1 + (consistency - 0.75)))
                + (ml_con_potential * 0.3)
                + (hist_con_score * 0.2)
            )
        circuit_feats = (
            self.circuit_model.get_circuit_features(circuit_id)
            if self.circuit_model.is_trained
            else {"avg_position_delta": 0.0, "avg_dnf_rate": 0.15}
        )
        raw_delta = circuit_feats.get("avg_position_delta", 0.0)
        overtaking_factor = float(np.clip(0.05 + (raw_delta / 20.0), 0.05, 0.60))
        weather_factor = self._get_weather_factor(
            weather_conditions, circuit_id, current_year
        )
        if weather_factor < 0.90:
            rain_intensity = 1.0 - weather_factor
            overtaking_factor = overtaking_factor * weather_factor
            uncertainty = np.random.uniform(-2, 2) * rain_intensity * 5
            base_score += uncertainty
        final_score = (grid_pos * overtaking_factor) + (base_score * (1 - overtaking_factor))
        final_position = float(np.clip(final_score, 1.0, 20.0))
        return {
            "driver_id": driver_id,
            "is_rookie": is_rookie,
            "expected_position_value": round(final_position, 2),
            "expected_position_str": f"P{final_position:.1f}",
            "factors": {
                "car_potential": round(float(ml_con_potential), 2),
                "pilot_performance": round(float(ml_pilot_pred), 2),
                "overtaking_difficulty": round(overtaking_factor, 2),
                "reliability_applied": round(reliability, 2),
                "consistency_applied": round(consistency, 2),
                "weather_factor": round(weather_factor, 2),
                "is_rainy": weather_factor < 0.90,
            },
        }
    def _get_current_constructor(self, driver_id: int, year: int) -> int:
        races_this_year = self.api.get_races_by_year(year)
        if not races_this_year.empty:
            last_race_id = (
                races_this_year
                .sort_values("round", ascending=False)
                .iloc[0]["raceId"]
            )
            quali = self.api.get_qualifying_by_race(last_race_id)
            if not quali.empty:
                driver_quali = quali[quali["driverId"] == driver_id]
                if not driver_quali.empty:
                    return int(driver_quali.iloc[0]["constructorId"])
        results = self.api.get_results_by_driver(driver_id)
        if not results.empty:
            return int(results.sort_values("raceId", ascending=False).iloc[0]["constructorId"])
        return 0

    def _get_last_race_id_before(self, year: int) -> int:
        races = self.api.get_all_races()
        if races.empty:
            return 0
        races["year"] = pd.to_numeric(races["year"], errors="coerce")
        past = races[races["year"] <= year]
        if past.empty:
            return int(races.sort_values("raceId").iloc[-1]["raceId"])
        return int(past.sort_values("raceId").iloc[-1]["raceId"])

    def _get_constructor_standings(
        self, constructor_id: int, year: int
    ) -> tuple[float, int, int]:
        race_id = self._get_last_race_id_before(year)
        con_std = self.api.get_constructor_standings_by_race(race_id)
        if not con_std.empty:
            row = con_std[con_std["constructorId"] == constructor_id]
            if not row.empty:
                return (
                    float(row.iloc[0]["points"]),
                    int(row.iloc[0]["wins"]),
                    int(row.iloc[0]["position"]),
                )
        return 0.0, 0, 10

    def _get_driver_points(self, driver_id: int, year: int) -> float:
        race_id = self._get_last_race_id_before(year)
        drv_std = self.api.get_driver_standings_by_race(race_id)
        if not drv_std.empty:
            row = drv_std[drv_std["driverId"] == driver_id]
            if not row.empty:
                return float(row.iloc[0]["points"])
        return 0.0

    def _get_reliability_score(self, constructor_id: int) -> float:
        results = self.api.get_results_by_constructor(constructor_id)
        if results.empty:
            return 0.85
        finished = (results["statusId"] == 1).sum()
        return round(float(finished / len(results)), 3)

    def _get_weather_factor(
        self, weather_conditions: dict, circuit_id: int, year: int
    ) -> float:
        if not self.weather_model.is_trained:
            rain_prob = weather_conditions.get("rain_prob", 0.0)
            return float(np.clip(1.0 - rain_prob * 0.4, 0.6, 1.0))
        rain_prob = weather_conditions.get("rain_prob", 0.0)
        races = self.api.get_races_by_year(year)
        month = 6
        if not races.empty and "date" in races.columns:
            try:
                month = int(
                    pd.to_datetime(races.iloc[0]["date"]).month
                )
            except Exception:
                pass
        circuit_feats = (
            self.circuit_model.get_circuit_features(circuit_id)
            if self.circuit_model.is_trained
            else {"avg_dnf_rate": 0.15}
        )
        weather_input = {
            "month": month,
            "rain_probability": float(np.clip(rain_prob, 0.0, 0.90)),
            "dnf_rate_delta": 0.0,
            "round": 8,
        }
        return self.weather_model.predict(weather_input)