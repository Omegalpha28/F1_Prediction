import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)

CLUSTER_LABELS = { 0: "street_circuit", 1: "high_speed", 2: "balanced", 3: "technical",}

class CircuitModel(Ml_Prediction):

    N_CLUSTERS = 4

    def __init__(self):
        super().__init__()
        self.model = KMeans(n_clusters=self.N_CLUSTERS, random_state=42, n_init=10)
        self._geo_features = ["lat", "lng", "alt"]
        self._derived_features = ["avg_position_delta", "avg_dnf_rate"]
        self.features = self._geo_features + self._derived_features
        self._circuit_clusters: dict[int, int] = {}
        self._circuit_scores: dict[int, dict] = {}

    def train(self, api, year: int = 9999) -> None:
        df_circuits, df_results, df_races = self._fetch_and_validate_data(api)
        df_merged = self._prepare_training_data(df_circuits, df_results, df_races, year)
        cluster_labels = self._fit_model(df_merged)
        self._store_circuit_scores(df_merged, cluster_labels)
        self.is_trained = True
        logger.info(f"[CircuitModel] KMeans entraîné — {len(self._circuit_clusters)} circuits indexés.")


    def _fetch_and_validate_data(self, api) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_circuits = api.get_all_circuits()
        df_results = api.get_all_results()
        df_races = api.get_all_races()
        if df_circuits.empty:
            raise ValueError("[CircuitModel] df_circuits est vide — vérifier F1API.")
        if df_results.empty:
            raise ValueError("[CircuitModel] df_results est vide — vérifier F1API.")
        return df_circuits, df_results, df_races

    def _prepare_training_data(self, df_circuits: pd.DataFrame, df_results: pd.DataFrame, df_races: pd.DataFrame, year: int) -> pd.DataFrame:
        df_derived = self._compute_derived_features(df_results, df_races, year)
        self._derived_features = ["avg_position_delta", "avg_dnf_rate", "overtaking_rate"]
        self.features = self._geo_features + self._derived_features
        df_merged = df_circuits.merge(df_derived, on="circuitId", how="left")
        for col in self._derived_features:
            median_val = df_merged[col].median()
            df_merged[col] = df_merged[col].fillna(median_val)
        return df_merged

    def _fit_model(self, df_merged: pd.DataFrame) -> np.ndarray:
        X = self.preprocess_features(df_merged, self.features)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self.model.labels_

    def _store_circuit_scores(self, df_merged: pd.DataFrame, cluster_labels: np.ndarray) -> None:
        self._circuit_clusters.clear()
        self._circuit_scores.clear()
        for idx, row in df_merged.iterrows():
            cid = int(row["circuitId"])
            cluster = int(cluster_labels[idx])
            self._circuit_clusters[cid] = cluster
            self._circuit_scores[cid] = {
                "cluster": cluster, "cluster_label": CLUSTER_LABELS.get(cluster, "unknown"),
                "avg_position_delta": float(row["avg_position_delta"]), "avg_dnf_rate": float(row["avg_dnf_rate"]),
                "overtaking_rate": float(row["overtaking_rate"]),
            }

    def _compute_overtaking_rate(self, df_results: pd.DataFrame, df_races: pd.DataFrame, year: int = 9999) -> pd.DataFrame:
        df_clean = self._prepare_overtaking_data(df_results, df_races, year)
        if df_clean.empty:
            return pd.DataFrame(columns=["circuitId", "overtaking_rate"])
        circuit_median = self._calculate_circuit_mobility(df_clean)
        circuit_median = self._normalize_overtaking_scores(circuit_median)
        return circuit_median[["circuitId", "overtaking_rate"]]

    def _prepare_overtaking_data(self, df_results: pd.DataFrame, df_races: pd.DataFrame, year: int) -> pd.DataFrame:
        df_races_filtered = df_races[df_races["year"] < year][["raceId", "circuitId", "year"]].copy()
        if df_races_filtered.empty:
            return pd.DataFrame()
        df = df_results.merge(df_races_filtered, on="raceId", how="inner")
        for col in ["grid", "positionOrder", "statusId"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = (df["statusId"] == 1) & (df["grid"] > 0) & df["grid"].notna() & df["positionOrder"].notna()
        df_clean = df[mask].copy()
        if not df_clean.empty:
            df_clean["positions_gained"] = df_clean["grid"] - df_clean["positionOrder"]
        return df_clean

    def _calculate_circuit_mobility(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        race_mobility = (
            df_clean.assign(moved=df_clean["positions_gained"] > 0)
            .groupby(["circuitId", "raceId"])
            .agg(
                movers=("moved", "sum"),
                classified=("moved", "count")
            )
            .assign(mobility=lambda x: x["movers"] / x["classified"])
            .reset_index()
        )
        return (
            race_mobility.groupby("circuitId")["mobility"]
            .median()
            .reset_index(name="overtaking_rate_raw")
        )

    def _normalize_overtaking_scores(self, circuit_median: pd.DataFrame) -> pd.DataFrame:
        min_val = circuit_median["overtaking_rate_raw"].min()
        max_val = circuit_median["overtaking_rate_raw"].max()
        if max_val == min_val:
            circuit_median["overtaking_rate"] = 0.5
        else:
            circuit_median["overtaking_rate"] = (
                (circuit_median["overtaking_rate_raw"] - min_val) / (max_val - min_val)
            )
        return circuit_median

    def _compute_derived_features(
        self, df_results: pd.DataFrame, df_races: pd.DataFrame, year: int = 9999,) -> pd.DataFrame:
        df_results = df_results.copy()
        df_results["grid"] = pd.to_numeric(df_results["grid"], errors="coerce")
        df_results["positionOrder"] = pd.to_numeric(
            df_results["positionOrder"], errors="coerce"
        )
        df = df_results.merge(
            df_races[["raceId", "circuitId"]], on="raceId", how="left"
        )
        df["position_delta"] = df["grid"] - df["positionOrder"]
        df["dnf"] = (df["statusId"] != 1).astype(int)
        agg = (
            df.groupby("circuitId")
            .agg(
                avg_position_delta=("position_delta", "mean"),
                avg_dnf_rate=("dnf", "mean"),
            )
            .reset_index()
        )
        df_overtaking = self._compute_overtaking_rate(df_results, df_races, year)
        agg = agg.merge(df_overtaking, on="circuitId", how="left")
        agg["overtaking_rate"] = agg["overtaking_rate"].fillna(
            agg["overtaking_rate"].median()
        )
        return agg

    def get_circuit_features(self, circuit_id: int) -> dict:
        if not self.is_trained:
            raise RuntimeError(
                "[CircuitModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )
        if circuit_id not in self._circuit_scores:
            logger.warning(
                f"[CircuitModel] circuit_id={circuit_id} inconnu — "
                "retour des valeurs médianes par défaut."
            )
            return {
                "cluster": -1, "cluster_label": "unknown", "avg_position_delta": 0.0,
                "avg_dnf_rate": 0.15, "overtaking_rate": 0.5,
            }
        return self._circuit_scores[circuit_id]

    def get_circuit_cluster_label(self, circuit_id: int) -> str:
        return self.get_circuit_features(circuit_id)["cluster_label"]

    def predict(self, circuit_data: dict) -> int:
        if not self.is_trained:
            raise RuntimeError(
                "[CircuitModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )
        df_input = pd.DataFrame([circuit_data])
        X = self.preprocess_features(df_input, self.features)
        X_scaled = self.scaler.transform(X)
        cluster = int(self.model.predict(X_scaled)[0])
        return cluster