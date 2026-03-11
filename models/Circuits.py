import pandas as pd
from sklearn.cluster import KMeans
from .Ml_Predictions import Ml_Prediction

class CircuitModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.features = ['lat', 'lng', 'alt']
        self.is_trained = False

    def train(self, df_circuits: pd.DataFrame):
        X = self.preprocess_features(df_circuits, self.features)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        print("[Circuits] K-Means entraîné sur les données géographiques.")

    def get_overtaking_factor(self, circuit_id: int) -> float:
        bloque_ids = [6, 15]
        if circuit_id in bloque_ids:
            return 0.90
        return 0.05

    def predict(self, circuit_data: dict) -> int:
        if not self.is_trained: return -1
        df_input = pd.DataFrame([circuit_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]