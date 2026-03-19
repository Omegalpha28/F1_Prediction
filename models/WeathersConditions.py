import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)


RAIN_PROBABILITY_BY_MONTH = {
    1: 0.15,
    2: 0.15,
    3: 0.20,
    4: 0.25,
    5: 0.20,
    6: 0.25,
    7: 0.20,
    8: 0.15,
    9: 0.25,
    10: 0.20,
    11: 0.15,
    12: 0.10,
}

RAINY_CIRCUIT_BONUS = {
    9:  0.20,
    13: 0.25,
    18: 0.15,
    22: 0.15,
    36: 0.10,
    67: 0.15,
}


class WeatherModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=60, max_depth=6, random_state=42
        )
        self.features = ["month", "rain_probability", "dnf_rate_delta", "round"]

        self._race_weather_factors: dict[int, float] = {}

    def train(self, api) -> None:
        df_races = api.get_all_races()
        df_results = api.get_all_results()

        if df_races.empty or df_results.empty:
            raise ValueError("[WeatherModel] Données F1API vides — vérifier F1API.")

        df = self._build_weather_dataset(df_races, df_results)

        if df.empty:
            raise ValueError("[WeatherModel] Dataset météo vide après construction.")
        df["weather_factor"] = np.clip(
            1.0
            - (df["rain_probability"] * 0.35)
            - (np.clip(df["dnf_rate_delta"], 0, 0.5) * 0.30),
            0.6,
            1.0,
        )

        X = self.preprocess_features(df, self.features)
        y = df["weather_factor"].values

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        df["predicted_factor"] = np.clip(
            self.model.predict(X_scaled), 0.6, 1.0
        )
        for _, row in df.iterrows():
            self._race_weather_factors[int(row["raceId"])] = float(
                row["predicted_factor"]
            )

        logger.info(
            f"[WeatherModel] Entraîné sur {len(df)} courses — "
            f"{len(self._race_weather_factors)} weather_factors indexés."
        )

    def _build_weather_dataset(
        self, df_races: pd.DataFrame, df_results: pd.DataFrame
    ) -> pd.DataFrame:
        df_races = df_races.copy()
        df_races["date"] = pd.to_datetime(df_races["date"], errors="coerce")
        df_races["month"] = df_races["date"].dt.month.fillna(6).astype(int)
        df_races["round"] = pd.to_numeric(df_races["round"], errors="coerce").fillna(1)
        df_results = df_results.copy()
        df_results["dnf"] = (df_results["statusId"] != 1).astype(int)
        dnf_per_race = (
            df_results.groupby("raceId")["dnf"]
            .mean()
            .reset_index()
            .rename(columns={"dnf": "dnf_rate"})
        )

        df = df_races[["raceId", "circuitId", "month", "round"]].merge(
            dnf_per_race, on="raceId", how="left"
        )
        df["dnf_rate"] = df["dnf_rate"].fillna(0.15)
        df_circuit_baseline = (
            df.groupby("circuitId")["dnf_rate"]
            .mean()
            .reset_index()
            .rename(columns={"dnf_rate": "avg_circuit_dnf"})
        )
        df = df.merge(df_circuit_baseline, on="circuitId", how="left")
        df["avg_circuit_dnf"] = df["avg_circuit_dnf"].fillna(0.15)
        df["dnf_rate_delta"] = df["dnf_rate"] - df["avg_circuit_dnf"]
        df["rain_probability"] = df["month"].map(RAIN_PROBABILITY_BY_MONTH).fillna(0.20)
        df["circuit_bonus"] = df["circuitId"].map(RAINY_CIRCUIT_BONUS).fillna(0.0)
        df["rain_probability"] = np.clip(
            df["rain_probability"] + df["circuit_bonus"], 0.0, 0.90
        )
        return df

    def predict_for_race(self, race_id: int) -> float:
        if not self.is_trained:
            raise RuntimeError(
                "[WeatherModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )
        if race_id not in self._race_weather_factors:
            logger.warning(
                f"[WeatherModel] raceId={race_id} inconnu — weather_factor par défaut : 1.0"
            )
            return 1.0
        return self._race_weather_factors[race_id]

    def predict(self, weather_data: dict) -> float:
        if not self.is_trained:
            raise RuntimeError(
                "[WeatherModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )

        defaults = {
            "month": 6,
            "rain_probability": 0.20,
            "dnf_rate_delta": 0.0,
            "round": 8,
        }
        for key, val in defaults.items():
            if key not in weather_data:
                logger.warning(
                    f"[WeatherModel] Feature '{key}' absente — valeur par défaut : {val}"
                )
                weather_data[key] = val

        df_input = pd.DataFrame([weather_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        prediction = float(self.model.predict(X_scaled)[0])
        return float(np.clip(prediction, 0.6, 1.0))