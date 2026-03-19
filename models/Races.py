import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)


class RaceModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.features = [
            "grid",
            "round",
            "year",
            "driver_points",
            "consistency",
            "aggression",
            "rolling_avg_position",
            "constructor_points",
            "reliability_score",
            "circuit_cluster",
            "avg_position_delta",
            "avg_dnf_rate",
            "weather_factor",
        ]

    def train(
        self,
        api,
        circuit_model=None,
        constructor_model=None,
        pilot_model=None,
        weather_model=None,
    ) -> None:
        df = api.get_training_matrix()
        if df.empty:
            raise ValueError("[RaceModel] Training matrix vide — vérifier F1API.")

        df = self._enrich_circuit_features(df, api, circuit_model)
        df = self._enrich_constructor_features(df, api, constructor_model)
        df = self._enrich_pilot_features(df, api, pilot_model)
        df = self._enrich_weather_features(df, weather_model)
        df = self._compute_rolling_avg_position(df, api)
        for col in self.features:
            if col not in df.columns:
                logger.warning(
                    f"[RaceModel] Feature '{col}' absente après enrichissement — "
                    "imputation par médiane globale."
                )
                df[col] = 10.0
            else:
                df[col] = df[col].fillna(df[col].median())
        if "positionOrder" not in df.columns:
            raise ValueError(
                "[RaceModel] Colonne 'positionOrder' absente du DataFrame."
            )

        X = self.preprocess_features(df, self.features)
        y = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(20)
        if "year" in df.columns:
            max_year = df["year"].max()
            train_mask = df["year"] < max_year
            X_train = X[train_mask]
            y_train = y[train_mask]
            logger.info(
                f"[RaceModel] Split temporel — train jusqu'à {max_year - 1} "
                f"({len(X_train)} lignes), test sur {max_year}."
            )
        else:
            X_train, y_train = X, y

        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        logger.info(
            f"[RaceModel] GradientBoosting entraîné — "
            f"{len(X_train)} lignes, {len(self.features)} features."
        )

    def _enrich_circuit_features(
        self, df: pd.DataFrame, api, circuit_model
    ) -> pd.DataFrame:
        if circuit_model is not None and circuit_model.is_trained:

            circuit_rows = []
            for cid in df["circuitId"].unique():
                feats = circuit_model.get_circuit_features(int(cid))
                feats["circuitId"] = cid
                circuit_rows.append(feats)

            df_circuit_feats = pd.DataFrame(circuit_rows).rename(
                columns={"cluster": "circuit_cluster"}
            )
            df = df.merge(df_circuit_feats, on="circuitId", how="left")
        else:
            logger.warning(
                "[RaceModel] CircuitModel absent — valeurs neutres pour features circuit."
            )
            df["circuit_cluster"] = 0
            df["avg_position_delta"] = 0.0
            df["avg_dnf_rate"] = 0.15

        return df

    def _enrich_constructor_features(
        self, df: pd.DataFrame, api, constructor_model
    ) -> pd.DataFrame:
        if constructor_model is not None and constructor_model.is_trained:
            df_results = api.get_all_results().copy()
            df_results["finished"] = (df_results["statusId"] == 1).astype(int)
            reliability = (
                df_results.groupby("constructorId")["finished"]
                .mean()
                .reset_index()
                .rename(columns={"finished": "reliability_score"})
            )
            df = df.merge(reliability, on="constructorId", how="left")
        else:
            logger.warning(
                "[RaceModel] ConstructorModel absent — reliability_score par défaut : 0.85"
            )
            df["reliability_score"] = 0.85

        df["reliability_score"] = df["reliability_score"].fillna(0.85)
        return df

    def _enrich_pilot_features(
        self, df: pd.DataFrame, api, pilot_model
    ) -> pd.DataFrame:
        if pilot_model is not None and pilot_model.is_trained:
            behavioral_rows = []
            for did in df["driverId"].unique():
                feats = pilot_model.get_driver_behavioral_features(int(did))
                feats["driverId"] = did
                behavioral_rows.append(feats)

            df_behavioral = pd.DataFrame(behavioral_rows)
            df = df.merge(df_behavioral, on="driverId", how="left")
        else:
            logger.warning(
                "[RaceModel] PilotModel absent — consistency=0.75, aggression=0.10"
            )
            df["consistency"] = 0.75
            df["aggression"] = 0.10

        df["consistency"] = df["consistency"].fillna(0.75)
        df["aggression"] = df["aggression"].fillna(0.10)
        return df

    def _enrich_weather_features(
        self, df: pd.DataFrame, weather_model
    ) -> pd.DataFrame:
        if weather_model is not None and weather_model.is_trained:
            df_races = df[["raceId"]].drop_duplicates()
            weather_factors = []
            for _, row in df_races.iterrows():
                factor = weather_model.predict_for_race(int(row["raceId"]))
                weather_factors.append(
                    {"raceId": row["raceId"], "weather_factor": factor}
                )
            df_weather = pd.DataFrame(weather_factors)
            df = df.merge(df_weather, on="raceId", how="left")
        else:
            logger.warning(
                "[RaceModel] WeatherModel absent — weather_factor par défaut : 1.0"
            )
            df["weather_factor"] = 1.0

        df["weather_factor"] = df["weather_factor"].fillna(1.0)
        return df

    def _compute_rolling_avg_position(
        self, df: pd.DataFrame, api
    ) -> pd.DataFrame:
        df_results = api.get_all_results()[["raceId", "driverId", "positionOrder"]].copy()
        df_races = api.get_all_races()[["raceId", "year", "round"]]

        df_hist = df_results.merge(df_races, on="raceId", how="left")
        df_hist["positionOrder"] = pd.to_numeric(
            df_hist["positionOrder"], errors="coerce"
        )
        df_hist = df_hist.sort_values(["driverId", "year", "round"])
        df_hist["rolling_avg_position"] = (
            df_hist.groupby("driverId")["positionOrder"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

        df_rolling = df_hist[["raceId", "driverId", "rolling_avg_position"]]
        df = df.merge(df_rolling, on=["raceId", "driverId"], how="left")
        df["rolling_avg_position"] = df["rolling_avg_position"].fillna(10.0)
        return df

    def predict(self, race_data: dict) -> float:
        if not self.is_trained:
            raise RuntimeError(
                "[RaceModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )

        defaults = {
            "grid": 10, "round": 1, "year": 2023,
            "driver_points": 0.0, "consistency": 0.75, "aggression": 0.10,
            "rolling_avg_position": 10.0,
            "constructor_points": 0.0, "reliability_score": 0.85,
            "circuit_cluster": 0, "avg_position_delta": 0.0, "avg_dnf_rate": 0.15,
            "weather_factor": 1.0,
        }
        for key, val in defaults.items():
            if key not in race_data:
                logger.warning(
                    f"[RaceModel] Feature '{key}' absente — valeur par défaut : {val}"
                )
                race_data[key] = val

        df_input = pd.DataFrame([race_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        prediction = float(self.model.predict(X_scaled)[0])
        return float(np.clip(prediction, 1.0, 20.0))