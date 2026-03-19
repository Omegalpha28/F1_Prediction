import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)


class ConstructorModel(Ml_Prediction):

    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=40, max_depth=8, random_state=42
        )
        self.features = [
            "constructor_points",
            "constructor_wins",
            "constructor_champ_pos",
            "reliability_score",
        ]

    def train(self, api) -> None:
        df = api.get_training_matrix()

        if df.empty:
            raise ValueError("[ConstructorModel] Training matrix vide — vérifier F1API.")
        if "reliability_score" not in df.columns:
            df = self._compute_reliability_score(df, api)
        if "positionOrder" not in df.columns:
            raise ValueError(
                "[ConstructorModel] Colonne 'positionOrder' absente du DataFrame."
            )

        X = self.preprocess_features(df, self.features)
        y = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(20)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        logger.info(
            f"[ConstructorModel] Entraîné sur {len(X)} lignes — "
            f"features : {self.features}"
        )

    def _compute_reliability_score(
        self, df: pd.DataFrame, api
    ) -> pd.DataFrame:
        df_results = api.get_all_results().copy()

        if df_results.empty or "constructorId" not in df_results.columns:
            logger.warning(
                "[ConstructorModel] Impossible de calculer reliability_score "
                "— colonne constructorId absente. Valeur par défaut : 0.85"
            )
            df["reliability_score"] = 0.85
            return df

        df_results["finished"] = (df_results["statusId"] == 1).astype(int)
        reliability = (
            df_results.groupby("constructorId")["finished"]
            .mean()
            .reset_index()
            .rename(columns={"finished": "reliability_score"})
        )
        df = df.merge(reliability, on="constructorId", how="left")
        df["reliability_score"] = df["reliability_score"].fillna(0.85)
        return df

    def predict(self, constructor_data: dict) -> float:
        if not self.is_trained:
            raise RuntimeError(
                "[ConstructorModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )
        defaults = {
            "constructor_points": 0.0,
            "constructor_wins": 0,
            "constructor_champ_pos": 10,
            "reliability_score": 0.85,
        }
        for key, val in defaults.items():
            constructor_data.setdefault(key, val)

        df_input = pd.DataFrame([constructor_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])