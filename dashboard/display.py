import streamlit as st
import pandas as pd
import plotly.express as px
from load_to_display import (
    load_data_and_models, get_available_races, get_races_for_year,
    build_race_labels, build_race_dataframe, run_simulations,
    compute_dnf_ids, compute_adjusted_ranks, build_display_rows,
    build_chart_df, build_factors_df, get_team_col, get_predicted_winner,
)

st.set_page_config(page_title="F1 Predictor Dashboard", page_icon="🏎️", layout="wide")


def render_header():
    st.title("🏎️ Explorateur & Prévisions Historiques F1")
    st.markdown("Ce dashboard permet d'explorer les données historiques F1 et de simuler les classements via les modèles ML entraînés.")
    st.markdown("---")


def render_sidebar(available_races: pd.DataFrame) -> tuple:
    st.sidebar.header("Paramètres de la Course")
    years       = sorted(available_races["year"].unique(), reverse=True)
    year        = st.sidebar.selectbox("📅 Année", years, index=0)
    races_year  = get_races_for_year(available_races, year)

    if races_year.empty:
        st.sidebar.warning("Aucune course avec résultats pour cette année.")
        return year, None, None, None

    labels      = build_race_labels(races_year)
    race_id     = st.sidebar.selectbox("🏁 Grand Prix", options=list(labels.keys()), format_func=lambda x: labels[x])
    weather_sim = render_weather_sidebar()
    return year, race_id, races_year[races_year["raceId"] == race_id].iloc[0], weather_sim


def render_weather_sidebar() -> dict:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌥️ Influence de la Météo")
    air_temp   = st.sidebar.slider("🌡️ Temp. Air (°C)", 10.0, 45.0, 25.0, 0.5)
    track_temp = st.sidebar.slider("🏎️ Temp. Piste (°C)", 15.0, 60.0, 32.0, 0.5)
    rain_prob  = st.sidebar.slider("🌧️ Risque Pluie (%)", 0, 100, 0, 5) / 100.0
    return {"air_temp": air_temp, "track_temp": track_temp, "rain_prob": rain_prob}


def render_race_info(selected_race: pd.Series):
    st.success(
        f"**GP :** {selected_race['name']} &nbsp;|&nbsp; "
        f"**Date :** {selected_race['date']} &nbsp;|&nbsp; "
        f"**Lieu :** {selected_race.get('name_circuit', '—')}, {selected_race.get('country', '—')}"
    )


def render_results_table(display_rows: list):
    st.subheader("📋 Résultats & Prédictions")
    st.dataframe(pd.DataFrame(display_rows), width='stretch')
    st.markdown("---")


def render_score_chart(chart_df: pd.DataFrame):
    st.subheader("📊 Score de Performance Estimé")
    fig = px.bar(chart_df, x="Pilote", y="Score ML", color="Écurie",
                 title="Score ML estimé par pilote (plus bas = meilleur)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, width='stretch')
    st.markdown("---")


def render_factors_chart(factors_df: pd.DataFrame):
    st.subheader("⚙️ Analyse Voiture vs Pilote")
    fig = px.scatter(
        factors_df, x="Note Pilote", y="Note Voiture", color="Écurie", text="Pilote",
        title="Matrice de Performance (idéal = bas gauche)",
        labels={"Note Pilote": "Performance Pilote Estimée", "Note Voiture": "Puissance Écurie Estimée"},
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, width='stretch')
    st.markdown("---")


def render_winners(df_full, adjusted_rank_map, sim_df, tcol):
    real_winner = df_full.iloc[0]
    pred_winner = get_predicted_winner(df_full, adjusted_rank_map, sim_df)
    col1, col2  = st.columns(2)

    col1.info(f"**🏆 Vainqueur Réel :** {real_winner['forename']} {real_winner['surname']} ({real_winner[tcol]})")

    if pred_winner is not None:
        col2.success(f"**🔮 Vainqueur Prédit (Après DNF) :** {pred_winner['forename']} {pred_winner['surname']} ({pred_winner[tcol]})")
    else:
        col2.warning("Vainqueur prédit introuvable dans les résultats.")


def main():
    render_header()

    with st.spinner("Chargement des données et entraînement des modèles…"):
        api, predictor = load_data_and_models()

    available_races             = get_available_races(api)
    year, race_id, selected_race, weather_sim = render_sidebar(available_races)

    if race_id is None:
        return

    render_race_info(selected_race)

    if not st.button("Lancer la Simulation 🚀", type="primary"):
        st.info("Configurez les paramètres dans la barre latérale puis cliquez sur **Lancer la Simulation**.")
        return

    with st.spinner("Simulation en cours…"):
        df_full = build_race_dataframe(api, race_id)
        if df_full.empty:
            st.warning("Aucun résultat historique trouvé pour cette course.")
            return

        predictions_map, sim_df = run_simulations(predictor, df_full, selected_race["circuitId"], year, weather_sim)
        rank_map                = dict(zip(sim_df["driverId"], sim_df["potential_rank"]))
        tcol                    = get_team_col(df_full)
        dnf_ids                 = compute_dnf_ids(df_full)
        adjusted_rank_map       = compute_adjusted_ranks(rank_map, dnf_ids)

    render_results_table(build_display_rows(df_full, predictions_map, rank_map, adjusted_rank_map, tcol))
    render_score_chart(build_chart_df(df_full, predictions_map, tcol))
    render_factors_chart(build_factors_df(df_full, predictions_map, tcol))
    render_winners(df_full, adjusted_rank_map, sim_df, tcol)


if __name__ == "__main__":
    main()