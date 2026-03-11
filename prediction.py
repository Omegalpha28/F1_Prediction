# prediction.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class F1Predictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=20, random_state=42)
        self.is_trained = False

    def train(self, df_results: pd.DataFrame):
        """Entraîne le modèle sur les données de résultats."""
        if df_results.empty:
            print("Erreur : Pas de données pour l'entraînement.")
            return
        df_clean = df_results[['grid', 'positionOrder']].dropna()
        X = df_clean[['grid']]
        y = df_clean['positionOrder']

        self.model.fit(X, y)
        self.is_trained = True
        print("Modèle ML entraîné avec succès !")

    def predict_position(self, grid_position: int) -> float:
        """Prédit la position finale en fonction de la position sur la grille."""
        if not self.is_trained:
            print("Erreur : Le modèle doit être entraîné avant de prédire.")
            return -1.0
        prediction = self.model.predict([[grid_position]])
        return prediction[0]