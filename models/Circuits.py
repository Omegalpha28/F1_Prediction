import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)

CLUSTER_LABELS = {
    0: "street_circuit",
    1: "high_speed",
    2: "balanced",
    3: "technical",
}


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

    def train(self, api) -> None:
        df_circuits = api.get_all_circuits()
        df_results = api.get_all_results()
        df_races = api.get_all_races()

        if df_circuits.empty:
            raise ValueError("[CircuitModel] df_circuits est vide — vérifier F1API.")
        if df_results.empty:
            raise ValueError("[CircuitModel] df_results est vide — vérifier F1API.")
        df_derived = self._compute_derived_features(df_results, df_races)
        df_merged = df_circuits.merge(df_derived, on="circuitId", how="left")
        for col in self._derived_features:
            median_val = df_merged[col].median()
            df_merged[col] = df_merged[col].fillna(median_val)
        X = self.preprocess_features(df_merged, self.features)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        cluster_labels = self.model.labels_
        for idx, row in df_merged.iterrows():
            cid = int(row["circuitId"])
            cluster = int(cluster_labels[idx])
            self._circuit_clusters[cid] = cluster
            self._circuit_scores[cid] = {
                "cluster": cluster,
                "cluster_label": CLUSTER_LABELS.get(cluster, "unknown"),
                "avg_position_delta": float(row["avg_position_delta"]),
                "avg_dnf_rate": float(row["avg_dnf_rate"]),
            }

        self.is_trained = True
        logger.info(
            f"[CircuitModel] KMeans entraîné — {len(self._circuit_clusters)} circuits indexés."
        )

    def _compute_derived_features(
        self, df_results: pd.DataFrame, df_races: pd.DataFrame
    ) -> pd.DataFrame:
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
                "cluster": -1,
                "cluster_label": "unknown",
                "avg_position_delta": 0.0,
                "avg_dnf_rate": 0.15,
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