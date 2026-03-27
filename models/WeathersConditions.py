import logging
import numpy as np
import pandas as pd
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)

class WeatherModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.is_trained = False
        self._race_weather_factors: dict[int, float] = {}

    def train(self, api) -> None:
        df_races = api.get_all_races()
        df_results = api.get_all_results()
        if df_races.empty or df_results.empty:
            raise ValueError("[WeatherModel] Données vides.")
        df_results["dnf"] = (df_results["statusId"] != 1).astype(int)
        dnf_per_race = df_results.groupby("raceId")["dnf"].mean().reset_index().rename(columns={"dnf": "dnf_rate"})
        df = df_races[["raceId", "circuitId"]].merge(dnf_per_race, on="raceId", how="left")
        df["dnf_rate"] = df["dnf_rate"].fillna(0.15)
        df_circuit_baseline = df.groupby("circuitId")["dnf_rate"].mean().reset_index().rename(columns={"dnf_rate": "avg_circuit_dnf"})
        df = df.merge(df_circuit_baseline, on="circuitId", how="left")
        df["dnf_rate_delta"] = df["dnf_rate"] - df["avg_circuit_dnf"]
        df["historical_weather_factor"] = np.clip(1.0 - (df["dnf_rate_delta"] * 1.5), 0.6, 1.0)
        for _, row in df.iterrows():
            self._race_weather_factors[int(row["raceId"])] = float(row["historical_weather_factor"])
        self.is_trained = True
        logger.info(f"[WeatherModel] Modèle Mathématique initialisé sur {len(self._race_weather_factors)} courses historiques.")

    def predict_for_race(self, race_id: int) -> float:
        if not self.is_trained:
            return 1.0
        return self._race_weather_factors.get(race_id, 1.0)

    def predict(self, weather_data: dict) -> float:
        air_temp = weather_data.get("air_temp", 25.0)
        track_temp = weather_data.get("track_temp", 35.0)
        rain_prob = weather_data.get("rain_prob", 0.0)
        rain_penalty = rain_prob * 0.30
        temp_diff = abs(track_temp - 35.0)
        grip_penalty = min(temp_diff / 40.0, 1.0) * 0.15
        delta_t = track_temp - air_temp
        delta_penalty = (1.0 - (1.0 / (1.0 + np.exp(-0.5 * (delta_t - 5.0))))) * 0.10
        weather_factor = 1.0 - rain_penalty - grip_penalty - delta_penalty
        return float(np.clip(weather_factor, 0.5, 1.0))
