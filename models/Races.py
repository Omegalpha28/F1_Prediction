import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from .Ml_Predictions import Ml_Prediction

class RaceModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeRegressor(max_depth=5, random_state=42)
        self.features = ['year', 'round']

    def train(self, df_races: pd.DataFrame):
        X = self.preprocess_features(df_races, self.features)
        y = X['round'] * 1.5
        self.model.fit(X, y)
        self.is_trained = True
        print("[Races] Arbre de décision entraîné sur le calendrier.")

    def predict(self, race_data: dict) -> float:
        if not self.is_trained: return 1.0
        df_input = pd.DataFrame([race_data])
        X = self.preprocess_features(df_input, self.features)
        return self.model.predict(X)[0]