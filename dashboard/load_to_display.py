import pandas as pd
import streamlit as st
from api import F1API
from prediction import F1Predictor


@st.cache_resource(show_spinner=False)
def load_data_and_models() -> tuple[F1API, F1Predictor]:
    api = F1API()
    predictor = F1Predictor(api)
    try:
        predictor.train_all()
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement des modèles : {e}")
        raise
    return api, predictor


def get_available_races(api: F1API) -> pd.DataFrame:
    races    = api.get_all_races()
    results  = api.get_all_results()
    circuits = api.get_all_circuits()

    race_ids = results["raceId"].unique()
    available = races[races["raceId"].isin(race_ids)].copy()
    return pd.merge(
        available,
        circuits[["circuitId", "name", "country"]],
        on="circuitId",
        how="left",
        suffixes=("", "_circuit"),
    )


def get_races_for_year(available_races: pd.DataFrame, year: int) -> pd.DataFrame:
    return available_races[available_races["year"] == year].sort_values("round")


def build_race_labels(races_for_year: pd.DataFrame) -> dict:
    return {
        row["raceId"]: (
            f"R{row['round']} — {row['name']} "
            f"({row.get('name_circuit', '')}, {row.get('country', '')})"
        )
        for _, row in races_for_year.iterrows()
    }


def build_race_dataframe(api: F1API, race_id: int) -> pd.DataFrame:
    results      = api.get_results_by_race(race_id)
    if results.empty:
        return pd.DataFrame()

    drivers      = api.get_all_drivers()
    constructors = api.get_all_constructors()
    status_codes = api.get_all_status()

    df = pd.merge(results, drivers, on="driverId", how="left", suffixes=("_res", "_driver"))
    df = pd.merge(df, constructors, on="constructorId", how="left", suffixes=("", "_con"))
    df = pd.merge(df, status_codes, on="statusId", how="left")

    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(999)
    df["grid"]          = pd.to_numeric(df["grid"], errors="coerce").fillna(10)
    return df.sort_values("positionOrder")


def run_simulations(
    predictor: F1Predictor,
    df_full: pd.DataFrame,
    circuit_id: int,
    year: int,
    weather_sim: dict,
) -> tuple[dict, pd.DataFrame]:
    predictions_map = {}
    sim_results     = []

    for _, row in df_full.iterrows():
        pred = _simulate_single(predictor, row, circuit_id, year, weather_sim)
        predictions_map[int(row["driverId"])] = pred
        sim_results.append({"driverId": int(row["driverId"]), "score": pred["expected_position_value"]})

    sim_df = (
        pd.DataFrame(sim_results)
        .sort_values("score")
        .reset_index(drop=True)
    )
    sim_df["potential_rank"] = range(1, len(sim_df) + 1)
    return predictions_map, sim_df


def _simulate_single(predictor, row, circuit_id, year, weather_sim) -> dict:
    try:
        return predictor.simulate_race(
            driver_id=int(row["driverId"]),
            circuit_id=int(circuit_id),
            grid_pos=int(row["grid"]),
            weather_conditions=weather_sim,
            current_year=year,
        )
    except Exception as e:
        return {
            "expected_position_value": 20.0,
            "expected_position_str": "N/A",
            "factors": {
                "car_potential": 10.0, "pilot_performance": 10.0,
                "overtaking_difficulty": 0.5, "reliability_applied": 0.85,
            },
            "is_rookie": False,
            "error": str(e),
        }


def compute_dnf_ids(df_full: pd.DataFrame) -> set:
    dnf_ids = set()
    for _, row in df_full.iterrows():
        status = str(row.get("status", ""))
        if status != "Finished" and "Lap" not in status:
            dnf_ids.add(int(row["driverId"]))
    return dnf_ids


def compute_adjusted_ranks(rank_map: dict, dnf_ids: set) -> dict:
    adjusted = {}
    for d_id, base_rank in rank_map.items():
        if d_id in dnf_ids:
            adjusted[d_id] = "❌ DNF"
        else:
            dnf_ahead      = sum(1 for c in dnf_ids if rank_map.get(c, 999) < base_rank)
            adjusted[d_id] = f"P{base_rank - dnf_ahead}"
    return adjusted


def build_display_rows(df_full, predictions_map, rank_map, adjusted_rank_map, tcol) -> list:
    rows = []
    for _, row in df_full.iterrows():
        d_id         = int(row["driverId"])
        pred         = predictions_map[d_id]
        adj_rank     = adjusted_rank_map.get(d_id, "—")
        status_label = str(row.get("status", ""))

        try:
            actual_pos = int(row["positionOrder"])
            result_str = f"P{actual_pos}" if (status_label == "Finished" or "Lap" in status_label) else status_label
        except (ValueError, TypeError):
            actual_pos, result_str = 999, status_label

        adj_rank_int  = int(adj_rank.replace("P", "")) if isinstance(adj_rank, str) and adj_rank.startswith("P") else 999
        accuracy_mark = "⭐" if actual_pos != 999 and adj_rank_int != 999 and abs(actual_pos - adj_rank_int) <= 1 else ""

        rows.append({
            "Pos Réelle":             result_str,
            "Pilote":                 f"{row['forename']} {row['surname']}",
            "Écurie":                 str(row[tcol])[:20],
            "Grille":                 f"P{int(row['grid'])}",
            "Score ML (Brut)":        pred["expected_position_str"],
            "Prédit (Initial)":       f"P{rank_map.get(d_id, '—')}",
            "Prédit (Après Abandons)": adj_rank,
            "Précision":              accuracy_mark,
            "Rookie":                 "🆕" if pred.get("is_rookie") else "",
        })
    return rows


def build_chart_df(df_full, predictions_map, tcol) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Pilote":   f"{r['forename']} {r['surname']}",
            "Écurie":   str(r[tcol]),
            "Score ML": predictions_map[int(r["driverId"])]["expected_position_value"],
        }
        for _, r in df_full.iterrows()
    ]).sort_values("Score ML")


def build_factors_df(df_full, predictions_map, tcol) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Pilote":       f"{r['forename']} {r['surname']}",
            "Écurie":       str(r[tcol]),
            "Note Voiture": predictions_map[int(r["driverId"])].get("factors", {}).get("car_potential", 10.0),
            "Note Pilote":  predictions_map[int(r["driverId"])].get("factors", {}).get("pilot_performance", 10.0),
        }
        for _, r in df_full.iterrows()
    ])


def get_team_col(df: pd.DataFrame) -> str:
    return "name_con" if "name_con" in df.columns else "name"


def get_predicted_winner(df_full, adjusted_rank_map, sim_df) -> pd.Series | None:
    valid = [d_id for d_id, r in adjusted_rank_map.items() if r == "P1"]
    winner_id = valid[0] if valid else int(sim_df.iloc[0]["driverId"])
    rows = df_full[df_full["driverId"] == winner_id]
    return rows.iloc[0] if not rows.empty else None