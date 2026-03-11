import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

class PilotModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42)
        self.features = ['grid', 'driver_points', 'constructor_points', 'consistency', 'aggression']

    def train(self, df_training_matrix: pd.DataFrame):
        for feat in ['consistency', 'aggression']:
            if feat not in df_training_matrix.columns:
                df_training_matrix[feat] = 0.8
        X = self.preprocess_features(df_training_matrix, self.features)
        y = df_training_matrix['positionOrder']
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("[Pilots] Entraîné avec comportement pilote (Consistency/Aggression).")

    def predict(self, pilot_data: dict) -> float:
        if not self.is_trained: return 20.0
        pilot_data.setdefault('consistency', 0.8)
        pilot_data.setdefault('aggression', 0.5)
        df_input = pd.DataFrame([pilot_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]