import logging
from abc import ABC, abstractmethod
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class Ml_Prediction(ABC):
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.features: list = []
        self._training_medians: dict = {}

    def preprocess_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        missing_cols = [f for f in features if f not in df.columns]
        if missing_cols:
            raise ValueError(
                f"[{self.__class__.__name__}] ERREUR CRITIQUE : Colonnes obligatoires "
                f"manquantes dans les données envoyées : {missing_cols}"
            )
        if not features:
            raise ValueError(
                f"[{self.__class__.__name__}] Aucune feature définie pour ce modèle."
            )
        df_clean = df[features].copy()
        df_clean = df_clean.replace(r"\\?N", np.nan, regex=True)
        df_clean = df_clean.apply(pd.to_numeric, errors="coerce")
        if not self.is_trained:
            self._training_medians = df_clean.median().to_dict()
            df_clean = df_clean.fillna(self._training_medians)
        else:
            df_clean = df_clean.fillna(self._training_medians)
            df_clean = df_clean.fillna(0)
        return df_clean.astype(float)

    def get_feature_names(self) -> list:
        return self.features

    def save_model(self, path: str) -> None:
        if not self.is_trained:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Impossible de sauvegarder : "
                "le modèle n'est pas encore entraîné."
            )
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "features": self.features,
            "training_medians": self._training_medians,
        }
        joblib.dump(payload, path)
        logger.info(f"[{self.__class__.__name__}] Modèle sauvegardé → {path}")

    def load_model(self, path: str) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.is_trained = payload["is_trained"]
        self.features = payload["features"]
        self._training_medians = payload.get("training_medians", {})
        logger.info(f"[{self.__class__.__name__}] Modèle chargé ← {path}")

    @abstractmethod
    def train(self, api) -> None:
        raise NotImplementedError("train() doit être définie dans la classe enfant.")

    @abstractmethod
    def predict(self, input_data) -> any:
        raise NotImplementedError("predict() doit être définie dans la classe enfant.")