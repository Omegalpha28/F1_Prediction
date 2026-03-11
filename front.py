# front.py
from api import F1API

def run_dashboard():
    print("="*50)
    print("🏁 BIENVENUE SUR LE DASHBOARD PROJECT VVVA 🏁")
    print("="*50)
    print("\n[1] Initialisation de l'API et chargement des CSV...")
    api = F1API()
    circuits = api.get_all_circuits()
    if not circuits.empty:
        print(f"-> Info : {len(circuits)} circuits trouvés dans la base de données.")
    print("\n[2] Initialisation du simulateur de Grand Prix...")
    api.setup_prediction_model()
    print("\n[3] Test de prédiction :")
    grid_pos_test = 1
    estimated_finish = api.simulate_race_result(grid_pos_test)
    if estimated_finish != -1:
        print(f"🏎️  SIMULATION : Un pilote partant {grid_pos_test}ème sur la grille...")
        print(f"🏆  Position finale estimée par l'algorithme : {estimated_finish:.1f}")
