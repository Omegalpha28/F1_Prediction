import streamlit as st
import pandas as pd
import plotly.express as px
from api import F1API
from prediction import F1Predictor

st.set_page_config(
    page_title="F1 Predictor Dashboard",
    page_icon="🏎️",
    layout="wide",
)

@st.cache_resource(show_spinner=False)
def load_data_and_models():
    api = F1API()
    predictor = F1Predictor(api)

    try:
        predictor.train_all()
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement des modèles : {e}")
        raise

    return api, predictor

def _build_race_dataframe(api: F1API, race_id: int) -> pd.DataFrame:
    results = api.get_results_by_race(race_id)
    if results.empty:
        return pd.DataFrame()

    drivers = api.get_all_drivers()
    constructors = api.get_all_constructors()
    status_codes = api.get_all_status()

    df = pd.merge(results, drivers, on="driverId", how="left", suffixes=("_res", "_driver"))
    df = pd.merge(df, constructors, on="constructorId", how="left", suffixes=("", "_con"))
    df = pd.merge(df, status_codes, on="statusId", how="left")

    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce").fillna(999)
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(10)

    return df.sort_values("positionOrder")


def _team_col(df: pd.DataFrame) -> str:
    return "name_con" if "name_con" in df.columns else "name"


def _run_simulations(
    predictor: F1Predictor,
    df_full: pd.DataFrame,
    circuit_id: int,
    year: int,
    weather_sim: dict,
) -> tuple[dict, pd.DataFrame]:
    predictions_map = {}
    sim_results = []

    for _, row in df_full.iterrows():
        try:
            pred = predictor.simulate_race(
                driver_id=int(row["driverId"]),
                circuit_id=int(circuit_id),
                grid_pos=int(row["grid"]),
                weather_conditions=weather_sim,
                current_year=year,
            )
        except Exception as e:
            pred = {
                "expected_position_value": 20.0,
                "expected_position_str": "N/A",
                "factors": {
                    "car_potential": 10.0,
                    "pilot_performance": 10.0,
                    "overtaking_difficulty": 0.5,
                    "reliability_applied": 0.85,
                    "consistency_applied": 0.8,
                },
                "is_rookie": False,
                "error": str(e),
            }

        predictions_map[int(row["driverId"])] = pred
        sim_results.append({
            "driverId": int(row["driverId"]),
            "score": pred["expected_position_value"],
        })

    sim_df = (
        pd.DataFrame(sim_results)
        .sort_values("score")
        .reset_index(drop=True)
    )
    sim_df["potential_rank"] = range(1, len(sim_df) + 1)
    return predictions_map, sim_df

def main():
    st.title("🏎️ Explorateur & Prévisions Historiques F1")
    st.markdown(
        "Ce dashboard permet d'explorer les données historiques F1 "
        "et de simuler les classements via les modèles ML entraînés."
    )
    st.markdown("---")

    # -- Chargement --
    with st.spinner("Chargement des données et entraînement des modèles… "
                    "(peut prendre quelques dizaines de secondes au premier lancement)"):
        api, predictor = load_data_and_models()

    # ------------------------------------------------------------------
    # Sidebar — sélection course + météo
    # ------------------------------------------------------------------
    st.sidebar.header("Paramètres de la Course")

    races = api.get_all_races()
    results_all = api.get_all_results()
    circuits = api.get_all_circuits()

    # Seules les courses avec des résultats sont proposées
    race_ids_with_results = results_all["raceId"].unique()
    available_races = races[races["raceId"].isin(race_ids_with_results)].copy()
    available_races = pd.merge(
        available_races,
        circuits[["circuitId", "name", "country"]],
        on="circuitId",
        how="left",
        suffixes=("", "_circuit"),
    )

    available_years = sorted(available_races["year"].unique(), reverse=True)
    year = st.sidebar.selectbox("📅 Année", available_years, index=0)

    races_for_year = available_races[
        available_races["year"] == year
    ].sort_values("round")

    if races_for_year.empty:
        st.sidebar.warning("Aucune course avec résultats pour cette année.")
        return

    race_labels = {
        row["raceId"]: f"R{row['round']} — {row['name']} ({row.get('name_circuit', '')},"
                       f" {row.get('country', '')})"
        for _, row in races_for_year.iterrows()
    }

    selected_race_id = st.sidebar.selectbox(
        "🏁 Grand Prix",
        options=list(race_labels.keys()),
        format_func=lambda x: race_labels[x],
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌥️ Influence de la Météo")
    air_temp = st.sidebar.slider("🌡️ Temp. Air (°C)", 10.0, 45.0, 25.0, 0.5)
    track_temp = st.sidebar.slider("🏎️ Temp. Piste (°C)", 15.0, 60.0, 32.0, 0.5)
    rain_prob = st.sidebar.slider("🌧️ Risque Pluie (%)", 0, 100, 0, 5) / 100.0
    weather_sim = {"air_temp": air_temp, "track_temp": track_temp, "rain_prob": rain_prob}
    selected_race = races_for_year[
        races_for_year["raceId"] == selected_race_id
    ].iloc[0]

    circuit_id = selected_race["circuitId"]
    circuit_name = selected_race.get("name_circuit", "—")
    country = selected_race.get("country", "—")

    st.success(
        f"**GP :** {selected_race['name']} &nbsp;|&nbsp; "
        f"**Date :** {selected_race['date']} &nbsp;|&nbsp; "
        f"**Lieu :** {circuit_name}, {country}"
    )
    if not st.button("Lancer la Simulation 🚀", type="primary"):
        st.info("Configurez les paramètres dans la barre latérale puis cliquez sur **Lancer la Simulation**.")
        return

    with st.spinner("Simulation en cours…"):
        df_full = _build_race_dataframe(api, selected_race_id)

        if df_full.empty:
            st.warning("Aucun résultat historique trouvé pour cette course.")
            return

        predictions_map, sim_df = _run_simulations(
            predictor, df_full, circuit_id, year, weather_sim
        )
        rank_map = dict(zip(sim_df["driverId"], sim_df["potential_rank"]))
        tcol = _team_col(df_full)
    st.subheader("📋 Résultats & Prédictions")

    display_data = []
    for _, row in df_full.iterrows():
        d_id = int(row["driverId"])
        pred = predictions_map[d_id]
        pot_rank = rank_map.get(d_id, "—")
        driver_name = f"{row['forename']} {row['surname']}"
        status_label = str(row.get("status", ""))

        try:
            actual_pos = int(row["positionOrder"])
            result_str = (
                f"P{actual_pos}"
                if (status_label == "Finished" or "Lap" in status_label)
                else status_label
            )
        except (ValueError, TypeError):
            actual_pos = 999
            result_str = status_label

        accuracy_mark = (
            "⭐" if actual_pos != 999 and abs(actual_pos - pot_rank) <= 1 else ""
        )

        display_data.append({
            "Pos Réelle": result_str,
            "Pilote": driver_name,
            "Écurie": str(row[tcol])[:20],
            "Grille": f"P{int(row['grid'])}",
            "Score ML": pred["expected_position_str"],
            "Classement Prédit": f"P{pot_rank}",
            "Précision": accuracy_mark,
            "Rookie": "🆕" if pred.get("is_rookie") else "",
        })

    st.dataframe(pd.DataFrame(display_data), width='stretch')
    st.markdown("---")
    st.subheader("📊 Score de Performance Estimé")

    chart_df = pd.DataFrame([
        {
            "Pilote": f"{r['forename']} {r['surname']}",
            "Écurie": str(r[tcol]),
            "Score ML": predictions_map[int(r["driverId"])]["expected_position_value"],
        }
        for _, r in df_full.iterrows()
    ]).sort_values("Score ML")

    fig1 = px.bar(
        chart_df,
        x="Pilote",
        y="Score ML",
        color="Écurie",
        title="Score ML estimé par pilote (plus bas = meilleur)",
    )
    fig1.update_yaxes(autorange="reversed")
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, width='stretch')
    st.markdown("---")
    st.subheader("⚙️ Analyse Voiture vs Pilote")

    factors_data = []
    for _, r in df_full.iterrows():
        d_id = int(r["driverId"])
        fac = predictions_map[d_id].get("factors", {})
        factors_data.append({
            "Pilote": f"{r['forename']} {r['surname']}",
            "Écurie": str(r[tcol]),
            "Note Voiture": fac.get("car_potential", 10.0),
            "Note Pilote": fac.get("pilot_performance", 10.0),
        })

    df_factors = pd.DataFrame(factors_data)
    fig2 = px.scatter(
        df_factors,
        x="Note Pilote",
        y="Note Voiture",
        color="Écurie",
        text="Pilote",
        title="Matrice de Performance (idéal = bas gauche)",
        labels={
            "Note Pilote": "Performance Pilote Estimée",
            "Note Voiture": "Puissance Écurie Estimée",
        },
    )
    fig2.update_traces(textposition="top center")
    st.plotly_chart(fig2, width='stretch')
    st.markdown("---")
    winner = df_full.iloc[0]
    pred_winner_id = int(sim_df.iloc[0]["driverId"])
    pred_winner_rows = df_full[df_full["driverId"] == pred_winner_id]

    col1, col2 = st.columns(2)
    col1.info(
        f"**🏆 Vainqueur Réel :** "
        f"{winner['forename']} {winner['surname']} ({winner[tcol]})"
    )

    if not pred_winner_rows.empty:
        pw = pred_winner_rows.iloc[0]
        col2.success(
            f"**🔮 Vainqueur Prédit :** "
            f"{pw['forename']} {pw['surname']} ({pw[tcol]})"
        )
    else:
        col2.warning("Vainqueur prédit introuvable dans les résultats.")


if __name__ == "__main__":
    main()