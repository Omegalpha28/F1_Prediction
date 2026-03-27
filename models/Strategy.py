import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)

class StrategyModel(Ml_Prediction):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=60,
            max_depth=8,
            random_state=42,
        )
        self.features = [
            "stop", "grid", "avg_pit_duration",
            "circuit_dnf_rate", "constructor_reliability", "round",
        ]

    def _load_raw_data(self, api) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_pit = api.get_all_pit_stops()
        df_results = api.get_all_results()
        df_races = api.get_all_races()
        if df_pit.empty:
            raise ValueError("[StrategyModel] df_pit_stops vide — vérifier F1API.")
        return df_pit.copy(), df_results, df_races

    def _cast_numeric_columns(self, df_pit: pd.DataFrame) -> pd.DataFrame:
        for col in ["lap", "stop", "milliseconds"]:
            df_pit[col] = pd.to_numeric(df_pit[col], errors="coerce")
        return df_pit

    def _add_avg_pit_duration(self, df_pit: pd.DataFrame) -> pd.DataFrame:
        avg_duration = (
            df_pit.groupby("driverId")["milliseconds"]
            .mean()
            .div(1000)
            .rename("avg_pit_duration")
        )
        return df_pit.merge(avg_duration, on="driverId", how="left")

    def _merge_race_metadata(self, df_pit: pd.DataFrame, df_races: pd.DataFrame) -> pd.DataFrame:
        return df_pit.merge(
            df_races[["raceId", "round", "circuitId", "year"]],
            on="raceId",
            how="left",
        )

    def _merge_grid_position(self, df_pit: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
        df_grid = df_results[["raceId", "driverId", "grid"]].copy()
        df_grid["grid"] = pd.to_numeric(df_grid["grid"], errors="coerce")
        return df_pit.merge(df_grid, on=["raceId", "driverId"], how="left")

    def _fill_missing_values(self, df_pit: pd.DataFrame) -> pd.DataFrame:
        df_pit["avg_pit_duration"] = df_pit["avg_pit_duration"].fillna(
            df_pit["avg_pit_duration"].median()
        )
        df_pit["circuit_dnf_rate"] = df_pit["circuit_dnf_rate"].fillna(0.15)
        df_pit["constructor_reliability"] = df_pit["constructor_reliability"].fillna(0.85)
        df_pit["grid"] = df_pit["grid"].fillna(10)
        df_pit["round"] = df_pit["round"].fillna(1)
        return df_pit.dropna(subset=["lap"])

    def _fit_model(self, df_pit: pd.DataFrame) -> None:
        X = self.preprocess_features(df_pit, self.features)
        y = df_pit["lap"].values
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info(
            f"[StrategyModel] Entraîné sur {len(X)} pit stops — features : {self.features}"
        )

    def _enrich_circuit_dnf_rate(
        self, df: pd.DataFrame, circuit_model
    ) -> pd.DataFrame:
        if circuit_model is not None and circuit_model.is_trained:
            circuit_dnf = {}
            for cid in df["circuitId"].unique():
                feats = circuit_model.get_circuit_features(int(cid))
                circuit_dnf[cid] = feats.get("avg_dnf_rate", 0.15)

            df["circuit_dnf_rate"] = df["circuitId"].map(circuit_dnf)
        else:
            logger.warning(
                "[StrategyModel] CircuitModel absent — circuit_dnf_rate par défaut : 0.15"
            )
            df["circuit_dnf_rate"] = 0.15
        return df

    def _enrich_constructor_reliability(
        self, df: pd.DataFrame, df_results: pd.DataFrame
    ) -> pd.DataFrame:
        if df_results.empty or "constructorId" not in df_results.columns:
            df["constructor_reliability"] = 0.85
            return df

        df_results = df_results.copy()
        df_results["finished"] = (df_results["statusId"] == 1).astype(int)
        reliability = (
            df_results.groupby(["raceId", "driverId"])["constructorId"]
            .first()
            .reset_index()
        )
        reliability = reliability.merge(
            df_results[["raceId", "driverId", "constructorId"]]
            .drop_duplicates()
            .merge(
                df_results.groupby("constructorId")["finished"]
                .mean()
                .reset_index()
                .rename(columns={"finished": "constructor_reliability"}),
                on="constructorId",
            ),
            on=["raceId", "driverId"],
            how="left",
        )

        df = df.merge(
            reliability[["raceId", "driverId", "constructor_reliability"]],
            on=["raceId", "driverId"],
            how="left",
        )
        return df

    def train(self, api, circuit_model=None) -> None:
        df_pit, df_results, df_races = self._load_raw_data(api)
        df_pit = self._cast_numeric_columns(df_pit)
        df_pit = self._add_avg_pit_duration(df_pit)
        df_pit = self._merge_race_metadata(df_pit, df_races)
        df_pit = self._enrich_circuit_dnf_rate(df_pit, circuit_model)
        df_pit = self._enrich_constructor_reliability(df_pit, df_results)
        df_pit = self._merge_grid_position(df_pit, df_results)
        df_pit = self._fill_missing_values(df_pit)
        self._fit_model(df_pit)

    def predict(self, strategy_data: dict) -> int:
        if not self.is_trained:
            raise RuntimeError(
                "[StrategyModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )
        defaults = {
            "stop": 1, "grid": 10, "avg_pit_duration": 25.0, "circuit_dnf_rate": 0.15,
            "constructor_reliability": 0.85, "round": 1,
        }
        for key, val in defaults.items():
            if key not in strategy_data:
                logger.warning(
                    f"[StrategyModel] Feature '{key}' absente — valeur par défaut : {val}"
                )
                strategy_data[key] = val
        df_input = pd.DataFrame([strategy_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        prediction = int(round(float(self.model.predict(X_scaled)[0])))
        return int(np.clip(prediction, 1, 70))
