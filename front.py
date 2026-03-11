import warnings
from API import F1API
from prediction import F1Predictor
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_f1_explorer():
    api = F1API()
    predictor = F1Predictor(api)
    predictor.train_all()
    print("\n" + "="*105)
    print("🏎️  EXPLORATEUR & PRÉDICTEUR DE DONNÉES HISTORIQUES F1 🏎️")
    print("="*105)

    try:
        year_input = input("\n📅 Entrez l'année (ex: 2024) : ")
        if not year_input: return
        year = int(year_input)
        circuit_query = input("📍 Entrez le nom du circuit ou de la ville : ")
    except ValueError:
        print("❌ Erreur : Veuillez entrer une année valide.")
        return

    circuits = api.get_all_circuits()
    found_circuit = circuits[
        circuits['name'].str.contains(circuit_query, case=False, na=False) | 
        circuits['location'].str.contains(circuit_query, case=False, na=False)
    ]

    if found_circuit.empty:
        print(f"❌ Aucun circuit trouvé pour : '{circuit_query}'")
        return

    circuit_id = found_circuit.iloc[0]['circuitId']
    circuit_name = found_circuit.iloc[0]['name']

    races = api.get_all_races()
    race_info = races[(races['year'] == year) & (races['circuitId'] == circuit_id)]

    if race_info.empty:
        print(f"❌ Pas de Grand Prix trouvé en {year} sur le circuit : {circuit_name}")
        return

    race_id = race_info.iloc[0]['raceId']
    race_date = race_info.iloc[0]['date']

    results = api.get_results_by_race(race_id)
    drivers = api.get_all_drivers()
    constructors = api.get_all_constructors()
    status_codes = api.get_all_status()

    df_full = pd.merge(results, drivers, on='driverId', how='left', suffixes=('_res', '_driver'))
    df_full = pd.merge(df_full, constructors, on='constructorId', how='left', suffixes=('', '_con'))
    df_full = pd.merge(df_full, status_codes, on='statusId', how='left')
    weather_sim = {"air_temp": 25.0, "track_temp": 32.0, "rain_prob": 0.0}
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
    print("\n" + "="*105)
    print(f"🏁 GP : {race_info.iloc[0]['name']} | DATE : {race_date}")
    print(f"📍 LIEU : {circuit_name}, {found_circuit.iloc[0]['country']}")
    print("="*105)
    header = f"{'POS':<4} | {'PILOTE':<20} | {'ÉCURIE':<15} | {'GRILLE':<6} | {'RÉEL':<12} | {'PRÉDICT° (VAL)':<15} | {'CLAST POTENTIEL'}"
    print(header)
    print("-" * 105)

    team_col = 'name_con' if 'name_con' in df_full.columns else 'name'

    for _, row in df_full.iterrows():
        d_id = row['driverId']
        pred = predictions_map[d_id]
        pot_rank = rank_map[d_id]
        driver_name = f"{row['forename']} {row['surname']}"
        grid_pos = f"P{int(row['grid'])}"
        status_label = str(row['status'])
        actual_pos_val = int(row['positionOrder'])
        result_str = f"P{actual_pos_val}" if (status_label == 'Finished' or 'Lap' in status_label) else status_label

        pred_score_str = pred['expected_position_str']
        potential_clast = f"P{pot_rank}"
        accuracy_mark = "⭐" if abs(actual_pos_val - pot_rank) <= 1 else ""

        print(f"{actual_pos_val:<4} | {driver_name:<20} | {str(row[team_col])[:15]:<15} | {grid_pos:<6} | {result_str:<12} | {pred_score_str:<15} | {potential_clast:<15} {accuracy_mark}")

    print("="*105)
    if not df_full.empty:
        winner = df_full.iloc[0]
        pred_winner_id = sim_df.iloc[0]['driverId']
        pred_winner_row = df_full[df_full['driverId'] == pred_winner_id].iloc[0]
        print(f"🏆 VAINQUEUR RÉEL : {winner['forename']} {winner['surname']} ({winner[team_col]})")
        print(f"🔮 VAINQUEUR PRÉDIT : {pred_winner_row['forename']} {pred_winner_row['surname']} ({pred_winner_row[team_col]})")

if __name__ == "__main__":
    run_f1_explorer()