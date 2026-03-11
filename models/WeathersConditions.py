import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

class WeatherModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=30, random_state=42)
        self.features = ['air_temp', 'track_temp', 'rain_prob']

    def train(self, df_dummy=None):
        np.random.seed(42)
        mock_data = pd.DataFrame({
            'air_temp': np.random.uniform(15, 35, 100),
            'track_temp': np.random.uniform(20, 50, 100),
            'rain_prob': np.random.uniform(0, 1, 100)
        })
        y = 1.0 - (mock_data['rain_prob'] * 0.4)
        self.model.fit(mock_data, y)
        self.is_trained = True
        print("[Weather] Modèle de stabilité météo entraîné (Factor: 0.6 - 1.0).")

    def predict(self, weather_data: dict) -> float:
        if not self.is_trained: return 1.0
        df_input = pd.DataFrame([weather_data])
        X = df_input[self.features]
        prediction = self.model.predict(X)[0]
        return np.clip(prediction, 0.6, 1.0)