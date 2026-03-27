import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)


class PilotModel(Ml_Prediction):
    INCIDENT_STATUS_IDS = {3, 4}

    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
        self.features = [
            "grid",
            "rolling_avg_position",
            "pos_std",
            "aggression",
            "circuit_cluster"
        ]
        self._driver_behavioral_features: dict[int, dict] = {}

    def train(self, api, circuit_model=None) -> None:
        df = api.get_training_matrix()
        if df.empty:
            raise ValueError("[PilotModel] Training matrix vide.")
        df = self._prepare_training_data(df, api, circuit_model)
        self._fit_model(df)
        self.is_trained = True
        logger.info(f"[PilotModel] Entraîné sur {len(df)} lignes — features : {self.features}")

    def _prepare_training_data(self, df: pd.DataFrame, api, circuit_model) -> pd.DataFrame:
        df_res = api.get_all_results()
        df_races = api.get_all_races()
        df_beh = self._compute_behavioral_features(df_res, df_races)
        self._save_latest_behavioral_stats(df_beh)
        df = df.merge(df_beh[["driverId", "raceId", "pos_std", "aggression"]], on=["driverId", "raceId"], how="left")
        df = self._enrich_rolling_avg_position(df, api)
        df = self._enrich_circuit_features(df, circuit_model)
        return self._fill_missing_values(df)

    def _compute_behavioral_features(self, df_res: pd.DataFrame, df_races: pd.DataFrame) -> pd.DataFrame:
        df = df_res.copy()
        df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
        df["statusId"] = pd.to_numeric(df["statusId"], errors="coerce")
        df["is_incident"] = df["statusId"].isin(self.INCIDENT_STATUS_IDS).astype(int)
        df = df.merge(df_races[["raceId", "year", "round"]], on="raceId", how="left").sort_values(["driverId", "year", "round"])
        grp = df.groupby("driverId")
        df["pos_std"] = grp["positionOrder"].transform(lambda x: x.shift(1).rolling(15, min_periods=1).std()).fillna(5.0)
        df["aggression"] = grp["is_incident"].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(0.0)
        return df[["driverId", "raceId", "pos_std", "aggression"]]

    def _save_latest_behavioral_stats(self, df_beh: pd.DataFrame) -> None:
        latest = df_beh.drop_duplicates(subset=["driverId"], keep="last")
        for _, row in latest.iterrows():
            self._driver_behavioral_features[int(row["driverId"])] = {
                "pos_std": float(row["pos_std"]),
                "aggression": float(row["aggression"]),
            }

    def _enrich_rolling_avg_position(self, df: pd.DataFrame, api) -> pd.DataFrame:
        df_res = api.get_all_results()[["raceId", "driverId", "positionOrder"]].copy()
        df_races = api.get_all_races()[["raceId", "year", "round"]]
        df_hist = df_res.merge(df_races, on="raceId", how="left")
        df_hist["positionOrder"] = pd.to_numeric(df_hist["positionOrder"], errors="coerce")
        df_hist = df_hist.sort_values(["driverId", "year", "round"])
        df_hist["rolling_avg_position"] = df_hist.groupby("driverId")["positionOrder"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        return df.merge(df_hist[["raceId", "driverId", "rolling_avg_position"]], on=["raceId", "driverId"], how="left")

    def _enrich_circuit_features(self, df: pd.DataFrame, circuit_model) -> pd.DataFrame:
        if circuit_model and circuit_model.is_trained:
            circuits = [{"circuitId": c, "circuit_cluster": circuit_model.get_circuit_features(int(c)).get("cluster", 0)} for c in df["circuitId"].unique()]
            return df.merge(pd.DataFrame(circuits), on="circuitId", how="left")
        df["circuit_cluster"] = 0
        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.features:
            df[col] = df[col].fillna(10.0 if col == "rolling_avg_position" else 0.0)
        return df

    def _fit_model(self, df: pd.DataFrame) -> None:
        if "positionOrder" not in df.columns:
            raise ValueError("[PilotModel] Colonne 'positionOrder' absente.")
        X = self.preprocess_features(df, self.features)
        y = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(20)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def get_driver_behavioral_features(self, driver_id: int) -> dict:
        if not self.is_trained:
            raise RuntimeError("[PilotModel] Non entraîné.")
        return self._driver_behavioral_features.get(driver_id, {"pos_std": 5.0, "aggression": 0.10})

    def predict(self, pilot_data: dict) -> float:
        if not self.is_trained:
            raise RuntimeError("[PilotModel] Non entraîné.")
        defaults = {"grid": 10, "pos_std": 5.0, "aggression": 0.10, "rolling_avg_position": 10.0, "circuit_cluster": 0}
        for key, val in defaults.items():
            pilot_data.setdefault(key, val)
        df_input = pd.DataFrame([pilot_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])
