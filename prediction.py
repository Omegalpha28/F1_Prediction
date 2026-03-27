import logging
import datetime
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
        self._train_model("CircuitModel", self.circuit_model.train, self.api)
        self._train_model("WeatherModel", self.weather_model.train, self.api)
        self._train_model("ConstructorModel", self.constructor_model.train, self.api, self.circuit_model)
        self._train_model("PilotModel", self.pilot_model.train, self.api, self.circuit_model)
        self._train_model("StrategyModel", self.strategy_model.train, self.api, self.circuit_model)
        self._train_model("RaceModel", self.race_model.train, self.api, self.circuit_model,
                          self.constructor_model, self.pilot_model, self.weather_model)
        self._log_training_results()

    def _log_training_results(self) -> None:
        trained = [name for name, m in self._all_models() if m.is_trained]
        failed = [name for name, m in self._all_models() if not m.is_trained]
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
            ("CircuitModel", self.circuit_model),
            ("WeatherModel", self.weather_model),
            ("ConstructorModel", self.constructor_model),
            ("PilotModel", self.pilot_model),
            ("StrategyModel", self.strategy_model),
            ("RaceModel", self.race_model),
        ]

    def simulate_race(self, driver_id: int, circuit_id: int, grid_pos: int, weather_conditions: dict, current_year: int = datetime.date.today().year) -> dict:
        context = self._fetch_race_context(driver_id, circuit_id, current_year)
        sub_feats = self._fetch_submodel_features(driver_id, circuit_id, weather_conditions, current_year)
        race_data = self._build_race_payload(grid_pos, current_year, context, sub_feats)
        final_pos, overt_factor = self._calculate_final_position(grid_pos, race_data, sub_feats)
        expected_dnfs = self.predict_expected_dnfs(circuit_id, sub_feats["weather_factor"])
        return self._build_simulation_report(driver_id, grid_pos, current_year, context, sub_feats, race_data, final_pos, overt_factor, expected_dnfs)

    def predict_expected_dnfs(self, circuit_id: int, weather_factor: float) -> int:
        circuit_feats = (
            self.circuit_model.get_circuit_features(circuit_id)
            if self.circuit_model.is_trained
            else {"avg_dnf_rate": self._get_global_avg_dnf_rate()} # Appel dynamique
        )
        base_dnf_rate = circuit_feats.get("avg_dnf_rate", self._get_global_avg_dnf_rate())
        weather_impact = (1.0 - weather_factor) * 0.40
        total_dnf_rate = base_dnf_rate + weather_impact
        total_dnf_rate = min(total_dnf_rate, 0.50)
        expected_dnfs = int(round(20 * total_dnf_rate))
        return expected_dnfs

    def _fetch_race_context(self, driver_id: int, circuit_id: int, current_year: int) -> dict:
        con_id = self._get_current_constructor(driver_id, current_year)
        con_pts, con_wins, con_pos = self._get_constructor_standings(con_id, current_year)
        roll_pts, dev_trend = self._get_constructor_form(con_id)
        return {
            "constructor_id": con_id,
            "constructor_points": con_pts,
            "constructor_wins": con_wins,
            "constructor_champ_pos": con_pos,
            "reliability": self._get_reliability_score(con_id),
            "current_round": self._get_round(circuit_id, current_year),
            "rolling_avg": self._get_rolling_avg_position(driver_id),
            "con_rolling_pts": roll_pts,
            "con_dev_trend": dev_trend
        }

    def _fetch_submodel_features(self, driver_id: int, circuit_id: int, weather: dict, year: int) -> dict:
        behavioral = self.pilot_model.get_driver_behavioral_features(driver_id) if self.pilot_model.is_trained else {"pos_std": 5.0, "aggression": 0.10}
        circuit = self.circuit_model.get_circuit_features(circuit_id) if self.circuit_model.is_trained else {"cluster": 0, "avg_position_delta": 0.0, "avg_dnf_rate": self._get_global_avg_dnf_rate(), "overtaking_rate": 0.5}
        weather_factor = self._get_weather_factor(weather, circuit_id, year)
        optimal_pit_lap = None
        if self.strategy_model.is_trained:
            constructor_id = self._get_current_constructor(driver_id, year)
            strategy_payload = {
                "stop": 1,
                "grid": self._get_rolling_avg_position(driver_id),
                "avg_pit_duration": self._get_avg_pit_duration(driver_id),
                "circuit_dnf_rate": circuit.get("avg_dnf_rate", self._get_global_avg_dnf_rate()),
                "constructor_reliability": self._get_reliability_score(constructor_id),
                "round": self._get_round(circuit_id, year),
            }
            optimal_pit_lap = self.strategy_model.predict(strategy_payload)
        return {
            "behavioral": behavioral, "circuit_feats": circuit, "weather_factor": weather_factor, "optimal_pit_lap": optimal_pit_lap,
        }

    def _build_race_payload(self, grid_pos: int, current_year: int, ctx: dict, sub: dict) -> dict:
        circ = sub["circuit_feats"]
        return {
            "grid": grid_pos,
            "pos_std": sub["behavioral"].get("pos_std", 5.0),
            "aggression": sub["behavioral"].get("aggression", 0.10),
            "rolling_avg_position": ctx["rolling_avg"],
            "reliability_score": ctx["reliability"],
            "circuit_cluster": circ.get("cluster", 0) if "cluster" in circ else circ.get("circuit_cluster", 0),
            "avg_position_delta": circ.get("avg_position_delta", 0.0),
            "avg_dnf_rate": circ.get("avg_dnf_rate", self._get_global_avg_dnf_rate()),
            "weather_factor": sub.get("weather_factor", 1.0),
            "optimal_pit_lap": sub.get("optimal_pit_lap") or self._get_avg_pit_lap(circuit_id=0), 
        }

    def _calculate_final_position(self, grid_pos: int, race_data: dict, sub: dict) -> tuple[float, float]:
        pred_pos = self.race_model.predict(race_data) if self.race_model.is_trained else float(grid_pos)
        overtaking_rate = sub["circuit_feats"].get("overtaking_rate", 0.5)
        overtaking_factor = float(np.clip(0.05 + (overtaking_rate * 0.30), 0.05, 0.35))
        if sub["weather_factor"] < 0.90:
            overtaking_factor *= sub["weather_factor"]
        final_score = (grid_pos * overtaking_factor) + (pred_pos * (1 - overtaking_factor))
        return float(np.clip(final_score, 1.0, 20.0)), overtaking_factor

    def _build_simulation_report(self, driver_id: int, grid_pos: int, year: int, ctx: dict, sub: dict, race_data: dict, final_pos: float, overt_factor: float, expected_dnfs: int) -> dict:
        con_data = {
            "constructor_points": ctx["constructor_points"], "constructor_wins": ctx["constructor_wins"], 
            "constructor_champ_pos": ctx["constructor_champ_pos"], "reliability_score": ctx["reliability"],
            "rolling_avg_points": ctx["con_rolling_pts"], "development_trend": ctx["con_dev_trend"],
            "circuit_cluster": sub["circuit_feats"].get("cluster", 0) if "cluster" in sub["circuit_feats"] else sub["circuit_feats"].get("circuit_cluster", 0)
        }
        ml_con = self.constructor_model.predict(con_data) if self.constructor_model.is_trained else 10.0
        pilot_data = {
            "grid": grid_pos,
            "pos_std": race_data["pos_std"],
            "aggression": race_data["aggression"],
            "rolling_avg_position": ctx["rolling_avg"],
            "circuit_cluster": con_data["circuit_cluster"]
        }
        ml_pilot = self.pilot_model.predict(pilot_data) if self.pilot_model.is_trained else 10.0
        ai_pred = self.race_model.predict(race_data) if self.race_model.is_trained else float(grid_pos)
        is_rookie = (self.api.get_driver_history_before(driver_id, year) is None)
        return {
            "driver_id": driver_id,
            "is_rookie": is_rookie,
            "expected_position_value": round(final_pos, 2),
            "expected_position_str": f"P{final_pos:.1f}",
            "factors": {
                "car_potential": round(float(ml_con), 2),
                "pilot_performance": round(float(ml_pilot), 2),
                "ai_race_prediction": round(float(ai_pred), 2),
                "overtaking_difficulty": round(overt_factor, 2),
                "reliability_applied": round(ctx["reliability"], 2),
                "weather_factor": round(sub["weather_factor"], 2),
                "is_rainy": sub["weather_factor"] < 0.90,
                "expected_dnfs": expected_dnfs,
                "optimal_pit_lap": sub.get("optimal_pit_lap"),
            },
        }

    def _get_avg_pit_duration(self, driver_id: int) -> float:
        df_pit = self.api.get_pit_stops_by_driver(driver_id).copy()
        if df_pit.empty:
            return self._get_global_avg_pit_duration()
        df_pit["milliseconds"] = pd.to_numeric(df_pit["milliseconds"], errors="coerce")
        avg_ms = df_pit["milliseconds"].mean()
        if pd.isna(avg_ms):
            return self._get_global_avg_pit_duration()
        return float(avg_ms / 1000.0)

    def _get_constructor_form(self, constructor_id: int) -> tuple[float, float]:
        results = self.api.get_results_by_constructor(constructor_id).copy()
        if results.empty:
            return 5.0, 0.0
        results["points"] = pd.to_numeric(results["points"], errors="coerce").fillna(0)
        grouped = results.groupby("raceId")["points"].sum().reset_index()
        grouped = grouped.sort_values("raceId", ascending=False)
        pts_history = grouped["points"].values
        if len(pts_history) == 0:
            return 5.0, 0.0
        short_term = float(np.mean(pts_history[:min(3, len(pts_history))]))
        if len(pts_history) < 10:
            long_term = short_term
            dev_trend = 0.0
        else:
            long_term = float(np.mean(pts_history[:10]))
            dev_trend = float(short_term - long_term)
        return short_term, dev_trend

    def _get_current_constructor(self, driver_id: int, year: int) -> int:
        races_this_year = self.api.get_races_by_year(year)
        if not races_this_year.empty:
            last_race_id = races_this_year.sort_values("round", ascending=False).iloc[0]["raceId"]
            quali = self.api.get_qualifying_by_race(last_race_id)
            if not quali.empty and not (driver_quali := quali[quali["driverId"] == driver_id]).empty:
                return int(driver_quali.iloc[0]["constructorId"])
        results = self.api.get_results_by_driver(driver_id)
        return int(results.sort_values("raceId", ascending=False).iloc[0]["constructorId"]) if not results.empty else 0

    def _get_last_race_id_before(self, year: int) -> int:
        races = self.api.get_all_races()
        if races.empty:
            return 0
        races["year"] = pd.to_numeric(races["year"], errors="coerce")
        past = races[races["year"] <= year]
        return int(past.sort_values("raceId").iloc[-1]["raceId"]) if not past.empty else int(races.sort_values("raceId").iloc[-1]["raceId"])

    def _get_constructor_standings(self, constructor_id: int, year: int) -> tuple[float, int, int]:
        race_id = self._get_last_race_id_before(year)
        con_std = self.api.get_constructor_standings_by_race(race_id)
        if not con_std.empty:
            row = con_std[con_std["constructorId"] == constructor_id]
            if not row.empty:
                return float(row.iloc[0]["points"]), int(row.iloc[0]["wins"]), int(row.iloc[0]["position"])
        return 0.0, 0, 10

    def _get_driver_points(self, driver_id: int, year: int) -> float:
        race_id = self._get_last_race_id_before(year)
        drv_std = self.api.get_driver_standings_by_race(race_id)
        if not drv_std.empty and not (row := drv_std[drv_std["driverId"] == driver_id]).empty:
            return float(row.iloc[0]["points"])
        return 0.0

    def _get_reliability_score(self, constructor_id: int) -> float:
        results = self.api.get_results_by_constructor(constructor_id)
        if results.empty:
            return self._get_global_reliability()
        finished = (results["statusId"] == 1).sum()
        return round(float(finished / len(results)), 3)

    def _get_weather_factor(self, weather_conditions: dict, circuit_id: int, year: int) -> float:
        if not self.weather_model.is_trained:
            rain_prob = weather_conditions.get("rain_prob", 0.0)
            return float(np.clip(1.0 - rain_prob * 0.4, 0.6, 1.0))
        weather_input = {
            "air_temp": float(weather_conditions.get("air_temp", 25.0)),
            "track_temp": float(weather_conditions.get("track_temp", 35.0)),
            "rain_prob": float(weather_conditions.get("rain_prob", 0.0))
        }
        return self.weather_model.predict(weather_input)

    def _extract_race_month(self, year: int) -> int:
        races = self.api.get_races_by_year(year)
        if not races.empty and "date" in races.columns:
            try:
                return int(pd.to_datetime(races.iloc[0]["date"]).month)
            except Exception: pass
        all_races = self.api.get_all_races()
        if not all_races.empty and "date" in all_races.columns:
             try:
                 return int(pd.to_datetime(all_races["date"]).dt.month.median())
             except Exception: pass
        return 6

    def _get_round(self, circuit_id: int, year: int) -> int:
        races = self.api.get_races_by_year(year)
        if not races.empty and not (match := races[races["circuitId"] == circuit_id]).empty:
            return int(match.iloc[0]["round"])

        if not races.empty:
            last_round = pd.to_numeric(races["round"], errors="coerce").max()
            return int(last_round + 1) if pd.notna(last_round) else 1
        return 1

    def _get_rolling_avg_position(self, driver_id: int) -> float:
        results = self.api.get_results_by_driver(driver_id)
        if results.empty:
            return 10.0
        last_5 = results.sort_values("raceId", ascending=False).head(5)
        positions = pd.to_numeric(last_5["positionOrder"], errors="coerce").dropna()
        return float(positions.mean()) if not positions.empty else 10.0

    def _get_global_avg_dnf_rate(self) -> float:
        results = self.api.get_all_results()
        if results.empty:
            return 0.15
        dnfs = (results["statusId"] != 1).sum()
        return float(dnfs / len(results))

    def _get_global_avg_pit_duration(self) -> float:
        df_pit = self.api.get_all_pit_stops()
        if df_pit.empty: return 23.5
        df_pit["milliseconds"] = pd.to_numeric(df_pit["milliseconds"], errors="coerce")
        avg_ms = df_pit["milliseconds"].mean()
        return float(avg_ms / 1000.0) if pd.notna(avg_ms) else 23.5

    def _get_global_reliability(self) -> float:
        results = self.api.get_all_results()
        if results.empty: return 0.85
        finished = (results["statusId"] == 1).sum()
        return float(finished / len(results))

    def _get_avg_pit_lap(self, circuit_id: int = 0) -> float:
        df_pit = self.api.get_all_pit_stops()
        if df_pit.empty: return 30.0
        df_pit["lap"] = pd.to_numeric(df_pit["lap"], errors="coerce")
        df_pit["stop"] = pd.to_numeric(df_pit["stop"], errors="coerce")
        first_stops = df_pit[df_pit["stop"] == 1]
        if first_stops.empty: return 30.0
        if circuit_id != 0:
            df_races = self.api.get_all_races()
            merged = first_stops.merge(df_races[["raceId", "circuitId"]], on="raceId")
            circuit_stops = merged[merged["circuitId"] == circuit_id]
            if not circuit_stops.empty:
                return float(circuit_stops["lap"].mean())
        return float(first_stops["lap"].mean())