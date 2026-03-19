import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .Ml_Predictions import Ml_Prediction

logger = logging.getLogger(__name__)


class StrategyModel(Ml_Prediction):
    """
    Modèle de prédiction de la stratégie de pit stop optimale en F1.

    Prédit le tour optimal pour effectuer un pit stop, par pilote et par course,
    en s'appuyant sur l'historique des arrêts et les caractéristiques de la course.

    Features utilisées :
      - stop              : numéro de l'arrêt (1er, 2ème, 3ème...)
      - grid              : position de départ (influence le moment du sous-cut/over-cut)
      - avg_pit_duration  : durée moyenne des pit stops du pilote (proxy fiabilité)
      - circuit_dnf_rate  : taux de DNF du circuit (influe sur la prudence stratégique)
      - constructor_reliability : fiabilité du constructeur (0-1)
      - round             : numéro de manche (fatigue pneus selon saison/règlement)

    Pourquoi ces features plutôt que ['raceId', 'stop'] ?
      - 'raceId' est un identifiant arbitraire, pas une feature métier.
        Le modèle ne peut pas généraliser sur un raceId inconnu.
      - 'stop' seul prédit juste que le 1er arrêt est tôt et le 2ème tard,
        ce qui est trivial et inutile.

    Usage type (orchestrateur) :
        strategy_model = StrategyModel()
        strategy_model.train(api, circuit_model)
        lap = strategy_model.predict({
            'stop': 1,
            'grid': 5,
            'avg_pit_duration': 23.5,
            'circuit_dnf_rate': 0.12,
            'constructor_reliability': 0.90,
            'round': 8,
        })
        # → 28  (tour optimal pour le 1er pit stop)
    """

    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=60,
            max_depth=8,
            random_state=42,
        )
        self.features = [
            "stop",
            "grid",
            "avg_pit_duration",
            "circuit_dnf_rate",
            "constructor_reliability",
            "round",
        ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, api, circuit_model=None) -> None:
        """
        Entraîne le modèle sur l'historique des pit stops enrichi.

        Étapes :
          1. Récupère pit_stops, results, races depuis F1API
          2. Calcule avg_pit_duration par pilote
          3. Enrichit avec circuit_dnf_rate depuis CircuitModel si disponible
          4. Fusionne constructor_reliability depuis results
          5. Entraîne RandomForest sur le tour réel du pit stop (target = lap)
        """
        df_pit = api.get_all_pit_stops()
        df_results = api.get_all_results()
        df_races = api.get_all_races()

        if df_pit.empty:
            raise ValueError("[StrategyModel] df_pit_stops vide — vérifier F1API.")

        df_pit = df_pit.copy()
        df_pit["lap"] = pd.to_numeric(df_pit["lap"], errors="coerce")
        df_pit["stop"] = pd.to_numeric(df_pit["stop"], errors="coerce")
        df_pit["milliseconds"] = pd.to_numeric(df_pit["milliseconds"], errors="coerce")

        # -- avg_pit_duration par pilote --
        avg_duration = (
            df_pit.groupby("driverId")["milliseconds"]
            .mean()
            .div(1000)  # ms → secondes
            .reset_index()
            .rename(columns={"milliseconds": "avg_pit_duration"})
        )
        df_pit = df_pit.merge(avg_duration, on="driverId", how="left")

        # -- Fusion avec races pour récupérer round et circuitId --
        df_pit = df_pit.merge(
            df_races[["raceId", "round", "circuitId", "year"]],
            on="raceId",
            how="left",
        )

        # -- circuit_dnf_rate depuis CircuitModel --
        df_pit = self._enrich_circuit_dnf_rate(df_pit, circuit_model)

        # -- constructor_reliability depuis results --
        df_pit = self._enrich_constructor_reliability(df_pit, df_results)

        # -- grid depuis results --
        df_grid = df_results[["raceId", "driverId", "grid"]].copy()
        df_grid["grid"] = pd.to_numeric(df_grid["grid"], errors="coerce")
        df_pit = df_pit.merge(df_grid, on=["raceId", "driverId"], how="left")

        # -- Imputation finale --
        df_pit["avg_pit_duration"] = df_pit["avg_pit_duration"].fillna(
            df_pit["avg_pit_duration"].median()
        )
        df_pit["circuit_dnf_rate"] = df_pit["circuit_dnf_rate"].fillna(0.15)
        df_pit["constructor_reliability"] = df_pit["constructor_reliability"].fillna(0.85)
        df_pit["grid"] = df_pit["grid"].fillna(10)
        df_pit["round"] = df_pit["round"].fillna(1)

        # -- Drop lignes sans target --
        df_pit = df_pit.dropna(subset=["lap"])

        X = self.preprocess_features(df_pit, self.features)
        y = df_pit["lap"].values

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        logger.info(
            f"[StrategyModel] Entraîné sur {len(X)} pit stops — "
            f"features : {self.features}"
        )

    # ------------------------------------------------------------------
    # Enrichissement features
    # ------------------------------------------------------------------

    def _enrich_circuit_dnf_rate(
        self, df: pd.DataFrame, circuit_model
    ) -> pd.DataFrame:
        """Ajoute circuit_dnf_rate depuis CircuitModel ou valeur par défaut."""
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
        """Calcule et fusionne constructor_reliability depuis results."""
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

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, strategy_data: dict) -> int:
        """
        Prédit le tour optimal pour un pit stop donné.

        Args:
            strategy_data: dict avec les clés de self.features.

        Returns:
            int : tour prédit pour le pit stop (clippé entre 1 et 70)
        """
        if not self.is_trained:
            raise RuntimeError(
                "[StrategyModel] Le modèle n'est pas entraîné. Appeler train() d'abord."
            )

        defaults = {
            "stop": 1,
            "grid": 10,
            "avg_pit_duration": 25.0,
            "circuit_dnf_rate": 0.15,
            "constructor_reliability": 0.85,
            "round": 1,
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