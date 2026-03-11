import pandas as pd
from sklearn.preprocessing import StandardScaler

class Ml_Prediction:
                                                                 
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def preprocess_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
                                                                               
        df_clean = df[features].copy()
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        df_clean = df_clean.replace(r'\\N', 0, regex=True)
        return df_clean.astype(float)

    def train(self, df: pd.DataFrame):
        raise NotImplementedError("La méthode train() doit être définie dans la classe enfant.")

    def predict(self, input_data):
        raise NotImplementedError("La méthode predict() doit être définie dans la classe enfant.")