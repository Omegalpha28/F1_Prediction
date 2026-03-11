import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

class ConstructorModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42)
        self.features = ['constructor_points', 'constructor_wins', 'constructor_champ_pos', 'reliability_score']

    def train(self, df_training_matrix: pd.DataFrame):
        if 'reliability_score' not in df_training_matrix.columns:
            df_training_matrix['reliability_score'] = 1.0
        X = self.preprocess_features(df_training_matrix, self.features)
        y = df_training_matrix['positionOrder']
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("[Constructors] Entraîné incluant le facteur fiabilité.")

    def predict(self, constructor_data: dict) -> float:
        if not self.is_trained: return 10.0
        constructor_data.setdefault('reliability_score', 0.95)
        df_input = pd.DataFrame([constructor_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]