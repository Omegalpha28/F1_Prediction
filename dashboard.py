import streamlit as st
import pandas as pd
import plotly.express as px
from api import F1API
from prediction import F1Predictor

st.set_page_config(page_title="F1 Predictor Dashboard", page_icon="🏎️", layout="wide")

@st.cache_resource
def load_data_and_models():
    api = F1API()
    predictor = F1Predictor(api)
    predictor.train_all()
    return api, predictor

def main():
    st.title("🏎️ Explorateur & Prévisions Historiques F1")
    st.markdown("Ce dashboard interactif permet d'explorer et de simuler les classements des courses de F1.")
    st.markdown("---")

    with st.spinner("Chargement des données et entraînement des modèles... (peut prendre quelques dizaines de secondes au premier lancement)"):
        api, predictor = load_data_and_models()

    # == SIDEBAR ==
    st.sidebar.header("Paramètres de la Course")
    
    # 1. Année
    year = st.sidebar.number_input("📅 Année", min_value=1950, max_value=2024, value=2024, step=1)
    
    # 2. Circuit
    circuit_query = st.sidebar.text_input("📍 Nom du circuit ou de ville", value="Monza")
    
    # Météo (Dynamique)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌥️ Influence de la Météo")
    air_temp = st.sidebar.slider("🌡️ Temp. Air (°C)", 10.0, 45.0, 25.0, 0.5)
    track_temp = st.sidebar.slider("🏎️ Temp. Piste (°C)", 15.0, 60.0, 32.0, 0.5)
    rain_prob = st.sidebar.slider("🌧️ Risque Pluie (%)", 0, 100, 0, 5) / 100.0
    
    weather_sim = {"air_temp": air_temp, "track_temp": track_temp, "rain_prob": rain_prob}

    # == MAIN CONTENT ==
    if not circuit_query:
        st.info("👈 Veuillez entrer un circuit dans le panneau latéral pour commencer.")
        return

    circuits = api.get_all_circuits()
    found_circuit = circuits[
        circuits['name'].str.contains(circuit_query, case=False, na=False) | 
        circuits['location'].str.contains(circuit_query, case=False, na=False)
    ]

    if found_circuit.empty:
        st.error(f"❌ Aucun circuit trouvé pour : '{circuit_query}'")
        return

    circuit_id = found_circuit.iloc[0]['circuitId']
    circuit_name = found_circuit.iloc[0]['name']
    country = found_circuit.iloc[0]['country']

    races = api.get_all_races()
    race_info = races[(races['year'] == year) & (races['circuitId'] == circuit_id)]

    if race_info.empty:
        st.warning(f"❌ Pas de Grand Prix trouvé en {year} sur le circuit : {circuit_name}")
        return

    race_id = race_info.iloc[0]['raceId']
    race_date = race_info.iloc[0]['date']
    race_name = race_info.iloc[0]['name']

    st.success(f"**GP :** {race_name} &nbsp;|&nbsp; **Date :** {race_date} &nbsp;|&nbsp; **Lieu :** {circuit_name}, {country}")
    
    if st.button("Lancer la Simulation 🚀", type="primary"):
        with st.spinner("Simulation en cours..."):
            results = api.get_results_by_race(race_id)
            if results.empty:
                st.warning("Aucun historique trouvé pour cette course.")
                return
            
            drivers = api.get_all_drivers()
            constructors = api.get_all_constructors()
            status_codes = api.get_all_status()

            # Fusion des données
            df_full = pd.merge(results, drivers, on='driverId', how='left', suffixes=('_res', '_driver'))
            df_full = pd.merge(df_full, constructors, on='constructorId', how='left', suffixes=('', '_con'))
            df_full = pd.merge(df_full, status_codes, on='statusId', how='left')
            
            predictions_map = {}
            sim_results = []

            for _, row in df_full.iterrows():
                pred = predictor.simulate_race(
                    driver_id=int(row['driverId']),
                    circuit_id=int(circuit_id),
                    grid_pos=int(row['grid']),
                    weather_conditions=weather_sim,
                    current_year=year
                )
                predictions_map[row['driverId']] = pred
                sim_results.append({
                    'driverId': row['driverId'],
                    'score': pred['expected_position_value']
                })

            sim_df = pd.DataFrame(sim_results).sort_values(by='score')
            sim_df['potential_rank'] = range(1, len(sim_df) + 1)
            rank_map = dict(zip(sim_df['driverId'], sim_df['potential_rank']))

            df_full = df_full.sort_values(by='positionOrder')

            display_data = []
            team_col = 'name_con' if 'name_con' in df_full.columns else 'name'

            for _, row in df_full.iterrows():
                d_id = row['driverId']
                pred = predictions_map[d_id]
                pot_rank = rank_map[d_id]
                driver_name = f"{row['forename']} {row['surname']}"
                grid_pos = f"P{int(row['grid'])}"
                
                status_label = str(row['status'])
                try:
                    actual_pos_val = int(row['positionOrder'])
                    result_str = f"P{actual_pos_val}" if (status_label == 'Finished' or 'Lap' in status_label) else status_label
                except (ValueError, TypeError):
                    actual_pos_val = 999
                    result_str = status_label

                pred_score_str = pred['expected_position_str']
                potential_clast = f"P{pot_rank}"
                accuracy_mark = "⭐" if actual_pos_val != 999 and abs(actual_pos_val - pot_rank) <= 1 else ""

                display_data.append({
                    "Pos Réelle": result_str,
                    "Pilote": driver_name,
                    "Écurie": str(row[team_col])[:15],
                    "Grille": grid_pos,
                    "Prédiction (Val)": pred_score_str,
                    "Class Potentiel": potential_clast,
                    "Précision": accuracy_mark
                })

            # Affichage du tableau
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Visualisation des Prédictions")
            
            # Graphique 1 : Score de performance
            sorted_sim_df = pd.DataFrame([
                {
                    'Pilote': f"{r['forename']} {r['surname']}",
                    'Écurie': r[team_col],
                    'Position Évaluée (Score ML)': predictions_map[r['driverId']]['expected_position_value']
                } for _, r in df_full.iterrows()
            ]).sort_values(by='Position Évaluée (Score ML)')
            
            fig1 = px.bar(sorted_sim_df, x='Pilote', y='Position Évaluée (Score ML)', color='Écurie',
                          title="Classement estimé (Plus le score est bas, meilleur est le pilote)")
            # Inverser l'axe Y pour avoir le 1er en haut
            fig1.update_yaxes(autorange="reversed")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("---")
            st.subheader("⚙️ Analyse de l'impact Voiture vs Pilote")
            
            # Graphique 2 : Voiture vs Pilote
            factors_data = []
            for _, r in df_full.iterrows():
                fac = predictions_map[r['driverId']]['factors']
                factors_data.append({
                    'Pilote': f"{r['forename']} {r['surname']}",
                    'Écurie': r[team_col],
                    'Note Voiture': fac['car_potential'],
                    'Note Pilote': fac['pilot_performance']
                })
                
            df_factors = pd.DataFrame(factors_data)
            fig2 = px.scatter(df_factors, x='Note Pilote', y='Note Voiture', color='Écurie',
                              text='Pilote', size_max=10, title="Matrice de Performance (Idéal = En bas à gauche)",
                              labels={'Note Pilote': 'Performance Pilote Estimée', 'Note Voiture': 'Puissance Écurie Estimée'})
            
            fig2.update_traces(textposition='top center')
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            
            # Affichage des vainqueurs
            if not df_full.empty:
                winner = df_full.iloc[0]
                pred_winner_id = sim_df.iloc[0]['driverId']
                pred_winner_row = df_full[df_full['driverId'] == pred_winner_id].iloc[0]
                
                col1, col2 = st.columns(2)
                col1.info(f"**🏆 Vainqueur Réel :** {winner['forename']} {winner['surname']} ({winner[team_col]})")
                col2.success(f"**🔮 Vainqueur Prédit :** {pred_winner_row['forename']} {pred_winner_row['surname']} ({pred_winner_row[team_col]})")

if __name__ == "__main__":
    main()
