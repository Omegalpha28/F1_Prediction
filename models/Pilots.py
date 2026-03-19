import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)


class PilotModel(Ml_Prediction):

    INCIDENT_STATUS_IDS = {3, 4}  # 3 = Accident, 4 = Collision

    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=60, max_depth=12, random_state=42
        )
        self.features = [
            "grid",
            "driver_points",
            "constructor_points",
            "consistency",
            "aggression",
        ]
        self._driver_behavioral_features: dict[int, dict] = {}

    def train(self, api) -> None:
        df = api.get_training_matrix()

        if df.empty:
            raise ValueError("[PilotModel] Training matrix vide — vérifier F1API.")
        df_results_full = api.get_all_results()
        df_behavioral = self._compute_behavioral_features(df_results_full)
        for _, row in df_behavioral.iterrows():
            self._driver_behavioral_features[int(row["driverId"])] = {
                "consistency": float(row["consistency"]),
                "aggression": float(row["aggression"]),
            }
        df = df.merge(df_behavioral, on="driverId", how="left")
        for col in ["consistency", "aggression"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        if "positionOrder" not in df.columns:
            raise ValueError(
                "[PilotModel] Colonne 'positionOrder' absente du DataFrame."
            )

        X = self.preprocess_features(df, self.features)
        y = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(20)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        logger.info(
            f"[PilotModel] Entraîné sur {len(X)} lignes — "
            f"features : {self.features}"
        )

    def _compute_behavioral_features(self, df_results: pd.DataFrame) -> pd.DataFrame:
        df = df_results.copy()
        df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
        df["statusId"] = pd.to_numeric(df["statusId"], errors="coerce")

        # -- Consistency --
        stats = (
            df.groupby("driverId")["positionOrder"]
            .agg(["mean", "std"])
            .reset_index()
        )
        stats.columns = ["driverId", "pos_mean", "pos_std"]
        stats["pos_std"] = stats["pos_std"].fillna(0)
        stats["consistency"] = np.where(
            stats["pos_mean"] > 0,
            1 - (stats["pos_std"] / stats["pos_mean"]),
            0.5,
        )
        stats["consistency"] = stats["consistency"].clip(0, 1)
        df["is_incident"] = df["statusId"].isin(self.INCIDENT_STATUS_IDS).astype(int)
        incident_rates = (
            df.groupby("driverId")["is_incident"]
            .mean()
            .reset_index()
            .rename(columns={"is_incident": "aggression"})
        )
        behavioral = stats[["driverId", "consistency"]].merge(
            incident_rates, on="driverId", how="left"
        )
        behavioral["aggression"] = behavioral["aggression"].fillna(0.0)

        return behavioral

    def get_driver_behavioral_features(self, driver_id: int) -> dict:
        if not self.is_trained:
            raise RuntimeError(
                "[PilotModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )
        if driver_id not in self._driver_behavioral_features:
            logger.warning(
                f"[PilotModel] driverId={driver_id} inconnu — "
                "retour des valeurs neutres par défaut."
            )
            return {"consistency": 0.75, "aggression": 0.10}
        return self._driver_behavioral_features[driver_id]

    def predict(self, pilot_data: dict) -> float:
        if not self.is_trained:
            raise RuntimeError(
                "[PilotModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )
        defaults = {
            "grid": 10,
            "driver_points": 0.0,
            "constructor_points": 0.0,
            "consistency": 0.75,
            "aggression": 0.10,
        }
        for key, val in defaults.items():
            if key not in pilot_data:
                logger.warning(
                    f"[PilotModel] Feature '{key}' absente de pilot_data — "
                    f"valeur par défaut utilisée : {val}"
                )
                pilot_data[key] = val

        df_input = pd.DataFrame([pilot_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])