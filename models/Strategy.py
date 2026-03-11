import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

class StrategyModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
        self.features = ['raceId', 'stop']

    def train(self, df_pit_stops: pd.DataFrame):
        if df_pit_stops.empty: return
        df_clean = df_pit_stops.copy()
        X = self.preprocess_features(df_clean, self.features)
        y = pd.to_numeric(df_clean['lap'], errors='coerce').fillna(20)
        self.model.fit(X, y)
        self.is_trained = True
        print("[Strategy] Modèle entraîné pour prédire le tour du Pit Stop.")

    def predict(self, strategy_data: dict) -> int:
        if not self.is_trained: return 20
        df_input = pd.DataFrame([strategy_data])
        X = self.preprocess_features(df_input, self.features)
        return int(self.model.predict(X)[0])