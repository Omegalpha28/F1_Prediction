import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)

class RaceModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self.features = [
            "grid", "pos_std", "aggression", "rolling_avg_position", "reliability_score", "circuit_cluster",
            "avg_position_delta", "avg_dnf_rate", "weather_factor",
        ]

    def train(self, api, circuit_model=None, constructor_model=None, pilot_model=None, weather_model=None) -> None:
        df = api.get_training_matrix()
        if df.empty:
            raise ValueError("[RaceModel] Training matrix vide.")
        df = self._enrich_circuit_features(df, api, circuit_model)
        df = self._enrich_constructor_features(df, api, constructor_model)
        df = self._enrich_pilot_features(df, api, pilot_model)
        df = self._enrich_weather_features(df, weather_model)
        df = self._compute_rolling_avg_position(df, api)
        if "positionOrder" not in df.columns:
            raise ValueError("[RaceModel] Colonne 'positionOrder' absente.")
        X = self.preprocess_features(df, self.features)
        y = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(20)
        if "year" in df.columns:
            max_year = df["year"].max()
            train_mask = df["year"] < max_year
            X_train = X[train_mask]
            y_train = y[train_mask]
        else:
            X_train, y_train = X, y
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        logger.info(f"[RaceModel] Entraîné sur {len(X_train)} lignes.")

    def _enrich_circuit_features(self, df: pd.DataFrame, api, circuit_model) -> pd.DataFrame:
        if circuit_model and circuit_model.is_trained:
            circuits = [{"circuitId": c, **circuit_model.get_circuit_features(int(c))} for c in df["circuitId"].unique()]
            df_circ = pd.DataFrame(circuits).rename(columns={"cluster": "circuit_cluster"})
            return df.merge(df_circ[["circuitId", "circuit_cluster", "avg_position_delta", "avg_dnf_rate"]], on="circuitId", how="left")
        df["circuit_cluster"] = 0
        df["avg_position_delta"] = 0.0
        df["avg_dnf_rate"] = 0.15
        return df

    def _enrich_constructor_features(self, df: pd.DataFrame, api, constructor_model) -> pd.DataFrame:
        df_res = api.get_all_results().copy()
        df_res["finished"] = (df_res["statusId"] == 1).astype(int)
        rel = df_res.groupby("constructorId")["finished"].mean().reset_index().rename(columns={"finished": "reliability_score"})
        return df.merge(rel, on="constructorId", how="left")

    def _enrich_pilot_features(self, df: pd.DataFrame, api, pilot_model) -> pd.DataFrame:
        if pilot_model and pilot_model.is_trained:
            beh = [{"driverId": d, **pilot_model.get_driver_behavioral_features(int(d))} for d in df["driverId"].unique()]
            return df.merge(pd.DataFrame(beh), on="driverId", how="left")
        df["pos_std"] = 5.0
        df["aggression"] = 0.10
        return df

    def _enrich_weather_features(self, df: pd.DataFrame, weather_model) -> pd.DataFrame:
        if weather_model and weather_model.is_trained:
            wf = [{"raceId": r, "weather_factor": weather_model.predict_for_race(int(r))} for r in df["raceId"].unique()]
            return df.merge(pd.DataFrame(wf), on="raceId", how="left")
        df["weather_factor"] = 1.0
        return df

    def _compute_rolling_avg_position(self, df: pd.DataFrame, api) -> pd.DataFrame:
        df_res = api.get_all_results()[["raceId", "driverId", "positionOrder"]].copy()
        df_races = api.get_all_races()[["raceId", "year", "round"]]
        df_hist = df_res.merge(df_races, on="raceId", how="left").sort_values(["driverId", "year", "round"])
        df_hist["positionOrder"] = pd.to_numeric(df_hist["positionOrder"], errors="coerce")
        df_hist["rolling_avg_position"] = df_hist.groupby("driverId")["positionOrder"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        return df.merge(df_hist[["raceId", "driverId", "rolling_avg_position"]], on=["raceId", "driverId"], how="left")

    def predict(self, race_data: dict) -> float:
        if not self.is_trained:
            raise RuntimeError("[RaceModel] Non entraîné.")
        defaults = {
            "grid": 10, "pos_std": 5.0, "aggression": 0.10, "rolling_avg_position": 10.0,
            "reliability_score": 0.85, "circuit_cluster": 0, "avg_position_delta": 0.0,
            "avg_dnf_rate": 0.15, "weather_factor": 1.0,
        }
        for key, val in defaults.items():
            race_data.setdefault(key, val)
        df_input = pd.DataFrame([race_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        prediction = float(self.model.predict(X_scaled)[0])
        return float(np.clip(prediction, 1.0, 20.0))