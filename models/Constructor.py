import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)

class ConstructorModel(Ml_Prediction):

    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42)
        self.features = [
            "constructor_points", "constructor_wins", "constructor_champ_pos",
            "reliability_score", "rolling_avg_points", "development_trend", "circuit_cluster"
        ]

    def train(self, api, circuit_model=None) -> None:
        df = api.get_training_matrix()
        if df.empty:
            raise ValueError("[ConstructorModel] Training matrix vide — vérifier F1API.")
        df = self._prepare_training_data(df, api, circuit_model)
        self._fit_model(df)
        self.is_trained = True

        logger.info(f"[ConstructorModel] Entraîné sur {len(df)} lignes — features : {self.features}")

    def _prepare_training_data(self, df: pd.DataFrame, api, circuit_model) -> pd.DataFrame:
        df = self._compute_reliability_score(df, api)
        df = self._enrich_constructor_form(df, api)
        df = self._enrich_circuit_features(df, circuit_model)
        return self._fill_missing_values(df)

    def _compute_reliability_score(self, df: pd.DataFrame, api) -> pd.DataFrame:
        df_results = api.get_all_results().copy()
        if df_results.empty or "constructorId" not in df_results.columns:
            df["reliability_score"] = 0.85
            return df
        df_results["finished"] = (df_results["statusId"] == 1).astype(int)
        rel = df_results.groupby("constructorId")["finished"].mean().reset_index()
        rel = rel.rename(columns={"finished": "reliability_score"})
        df = df.merge(rel, on="constructorId", how="left")
        df["reliability_score"] = df["reliability_score"].fillna(0.85)
        return df

    def _enrich_constructor_form(self, df: pd.DataFrame, api) -> pd.DataFrame:
        df_res = api.get_all_results()[["raceId", "constructorId", "points"]].copy()
        df_races = api.get_all_races()[["raceId", "year", "round"]]
        df_res["points"] = pd.to_numeric(df_res["points"], errors="coerce").fillna(0)
        df_grouped = df_res.groupby(["raceId", "constructorId"])["points"].sum().reset_index()
        df_hist = df_grouped.merge(df_races, on="raceId").sort_values(["constructorId", "year", "round"])
        grp = df_hist.groupby("constructorId")["points"]
        df_hist["rolling_avg_points"] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        short_term = grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        long_term = grp.transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        df_hist["development_trend"] = short_term - long_term
        feats = df_hist[["raceId", "constructorId", "rolling_avg_points", "development_trend"]]
        return df.merge(feats, on=["raceId", "constructorId"], how="left")

    def _enrich_circuit_features(self, df: pd.DataFrame, circuit_model) -> pd.DataFrame:
        if circuit_model and circuit_model.is_trained:
            circuits = [
                {"circuitId": cid, "circuit_cluster": circuit_model.get_circuit_features(int(cid)).get("cluster", 0)} 
                for cid in df["circuitId"].unique()
            ]
            return df.merge(pd.DataFrame(circuits), on="circuitId", how="left")
        df["circuit_cluster"] = 0
        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)
        return df

    def _fit_model(self, df: pd.DataFrame) -> None:
        if "positionOrder" not in df.columns:
            raise ValueError("[ConstructorModel] Colonne 'positionOrder' absente.")
        X = self.preprocess_features(df, self.features)
        y = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(20)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, constructor_data: dict) -> float:
        if not self.is_trained:
            raise RuntimeError("[ConstructorModel] Non entraîné.")
        defaults = {
            "constructor_points": 0.0, "constructor_wins": 0, "constructor_champ_pos": 10,
            "reliability_score": 0.85, "rolling_avg_points": 5.0, "development_trend": 0.0, 
            "circuit_cluster": 0
        }
        for key, val in defaults.items():
            constructor_data.setdefault(key, val)
        df_input = pd.DataFrame([constructor_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])