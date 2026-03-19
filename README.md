# F1_Detector

Create a realistic F1 race prediction model with Python.

## Prerequisites
- Python 3.13+ (or 3.12/3.11 with adjustment)
- `make` (Linux/macOS/WSL or Git Bash on Windows)
- `pip` and `virtualenv` (installed with Python)

## Installation and setup
1. Clone the repository:
   ```bash
   git clone <url> && cd F1_Detector
   ```
2. Create and install the virtual environment:
   ```bash
   make install
   ```

> This make target creates `.venv`, upgrades pip/setuptools/wheel, then installs `requirement.txt`.

## Run
- Without explicitly activating the venv:
  ```bash
  make run
  ```
- Directly with venv Python:
  ```bash
  ./.venv/bin/python main.py
  ```
- On Windows (PowerShell):
  ```powershell
  .\.venv\Scripts\Activate.ps1
  python main.py
  ```

## Useful Make commands
- `make`: install + build
- `make venv`: create `.venv` if absent
- `make install`: install dependencies
- `make run`: run the main script (`main.py`)
- `make build`: create a wrapper executable `F1_Detector`
- `make test`: check library imports
- `make clean`: remove executable and Python caches
- `make fclean`: `clean` + remove `.venv`
- `make re`: `fclean` then `all`

## CSV data handling
- Source data is in `stats/`.
- `main.py` includes functions:
  - `data()` to load all CSV files
  - `get_data(table)` to get a table
  - `get_row(table, idx)` to get a row
  - `set_value(table, idx, field, value)` to modify a value in memory
  - `print_table(table, limit)` to display rows

## Note
To ensure the project uses environment Python and packages (instead of system ones), activate the venv or run with `./.venv/bin/python`.

## API Getter methods (for devs)

In the file `api.py`, the class named `F1API` use some getters to access at the database in `stats/*` :

- Globals Methods
  - `get_table(table_name)`
  - `get_table_by(table_name, **filters)`

- Drivers
  - `get_all_drivers()`
  - `get_driver_info(driver_ref)`
  - `get_driver_by_id(driver_id)`
  - `get_driver_by_ref(driver_ref)`
  - `get_drivers_by_nationality(nationality)`
  - `get_drivers_by_name(forename=None, surname=None)`

- Circuits
  - `get_all_circuits()`
  - `get_circuit_by_id(circuit_id)`
  - `get_circuit_by_ref(circuit_ref)`
  - `get_circuits_by_country(country)`

- Constructors
  - `get_all_constructor_results()`
  - `get_all_constructor_standings()`
  - `get_constructor_by_id(constructor_id)`
  - `get_constructor_by_ref(constructor_ref)`
  - `get_constructors_by_nationality(nationality)`

- Races
  - `get_all_races()`
  - `get_race_by_id(race_id)`
  - `get_races_by_year(year)`
  - `get_races_by_circuit(circuit_id)`
  - `get_races_by_name(name)`

- Results
  - `get_all_results()`
  - `get_result_by_id(result_id)`
  - `get_results_by_race(race_id)`
  - `get_results_by_driver(driver_id)`
  - `get_results_by_constructor(constructor_id)`
  - `get_results_by_status(status_id)`

- Sprint Results
  - `get_all_sprint_results()`
  - `get_sprint_results_by_race(race_id)`
  - `get_sprint_results_by_driver(driver_id)`

- Constructor Results
  - `get_constructor_results_by_race(race_id)`
  - `get_constructor_results_by_constructor(constructor_id)`

- Standings
  - `get_all_driver_standings()`
  - `get_driver_standings_by_race(race_id)`
  - `get_driver_standings_by_driver(driver_id)`
  - `get_all_constructor_standings()`
  - `get_constructor_standings_by_race(race_id)`
  - `get_constructor_standings_by_constructor(constructor_id)`

- Lap/Pit/Qualifying
  - `get_all_lap_times()`
  - `get_lap_times_by_race(race_id)`
  - `get_lap_times_by_driver(driver_id)`
  - `get_all_pit_stops()`
  - `get_pit_stops_by_race(race_id)`
  - `get_pit_stops_by_driver(driver_id)`
  - `get_all_qualifying()`
  - `get_qualifying_by_race(race_id)`
  - `get_qualifying_by_driver(driver_id)`
  - `get_qualifying_by_constructor(constructor_id)`

- Seasons/Status
  - `get_all_seasons()`
  - `get_season_by_year(year)`
  - `get_all_status()`
  - `get_status_by_id(status_id)`


### Example
```python
from api import F1API
api = F1API()
drivers = api.get_all_drivers()
hamilton = api.get_driver_by_ref('hamilton')
results_2019 = api.get_results_by_race(1)
```

## Architecture ML

Le système de prédiction est organisé en une hiérarchie de classes ML. Chaque modèle hérite d'une classe abstraite commune et est spécialisé pour une tâche précise.

---

### `models/Ml_Prediction.py` — Classe de base abstraite

Classe abstraite dont héritent tous les modèles ML du projet. Elle centralise les comportements communs pour éviter la duplication de code entre les modèles enfants.

**Ce qu'elle fournit à tous les enfants :**

- **`preprocess_features(df, features)`** — Nettoie un DataFrame avant entraînement ou prédiction. Gère les valeurs manquantes (`\N` du format Ergast, `NaN`), valide que les colonnes demandées existent, et caste tout en `float`. L'ordre est important : remplacement des `\N` en `NaN` d'abord, puis `fillna` sur la médiane, puis cast.

- **`get_feature_names()`** — Retourne la liste des features consommées par le modèle. Utilisé par l'orchestrateur (`F1Predictor`) pour savoir quelles colonnes chaque modèle attend.

- **`save_model(path)` / `load_model(path)`** — Persistance via `joblib`. Sauvegarde et recharge le modèle entraîné, le scaler et la liste de features dans un seul fichier. Commun à tous les enfants, aucun ne doit surcharger ces méthodes.

**Ce que chaque enfant doit implémenter :**

- **`train(api)`** — Entraînement du modèle. Reçoit l'instance `F1API` pour récupérer lui-même les données dont il a besoin.
- **`predict(input_data)`** — Prédiction à partir d'une entrée. Le type de `input_data` dépend du modèle enfant (dict, DataFrame, etc.).

**Pourquoi `ABC` plutôt que `raise NotImplementedError` nu ?**
Avec `ABC`, Python refuse d'instancier `Ml_Prediction` directement et lève une erreur immédiate si un enfant oublie d'implémenter `train` ou `predict`. L'ancienne approche avec `raise NotImplementedError` laissait instancier la classe sans erreur jusqu'à l'appel de la méthode.

###odels/Circuits.py — Classification des circuits

Ce modèle répond à deux questions : **quel type de circuit est-ce ?** et **est-ce facile de dépasser dessus ?**

Il utilise un algorithme de clustering (KMeans) pour regrouper automatiquement les circuits en 4 familles selon leurs caractéristiques historiques et géographiques. Ces informations sont ensuite réutilisées par les autres modèles comme contexte supplémentaire.

**Ce qu'il calcule à l'entraînement :**

- **Le type de circuit** — en combinant la position GPS du circuit (latitude, longitude, altitude) avec deux métriques calculées depuis l'historique des résultats. Chaque circuit se voit attribuer un label parmi `street_circuit`, `high_speed`, `balanced` ou `technical`.

- **`avg_position_delta`** — la moyenne des places gagnées ou perdues par les pilotes sur ce circuit (différence entre position de départ et position d'arrivée). Une valeur élevée signifie que les dépassements sont fréquents.

- **`avg_dnf_rate`** — le taux de pilotes qui n'ont pas terminé la course sur ce circuit historiquement. Proxy de la dangerosité ou de la difficulté mécanique du tracé.

**Ce que les autres modèles consomment :**

Via `get_circuit_features(circuit_id)`, l'orchestrateur récupère un dictionnaire complet pour chaque circuit et l'injecte dans `RaceModel` et `StrategyModel` comme features supplémentaires.

---

### `models/Constructor.py` — Performance des écuries

Ce modèle prédit la position de course attendue pour une écurie donnée, en s'appuyant sur ses statistiques de championnat et sa fiabilité mécanique historique.

Il utilise une Random Forest entraînée sur l'historique complet des résultats fusionné avec les standings constructeurs.

**Features utilisées :**

- `constructor_points` — points au championnat à l'instant T de la course
- `constructor_wins` — nombre de victoires sur la saison en cours
- `constructor_champ_pos` — position au classement constructeurs
- `reliability_score` — taux de courses terminées par l'écurie, calculé automatiquement depuis l'historique des résultats (nombre de `statusId == 1` divisé par le total de courses disputées)

**Pourquoi `reliability_score` est calculé et non hardcodé ?**

L'ancienne version fixait cette valeur à `1.0` pour tout le monde, ce qui rendait la feature inutile pour le modèle. Maintenant chaque écurie a son propre score basé sur son historique réel — Red Bull et Mercedes auront naturellement un score plus élevé que des écuries plus fragiles.

**Sortie :** un float entre 1.0 et 20.0 représentant la position estimée. Utilisé par `F1Predictor` comme composante du score final dans `simulate_race()`.

### `models/Pilots.py` — Performance des pilotes

Ce modèle prédit la position de course attendue pour un pilote donné, en s'appuyant sur ses statistiques de championnat et deux métriques comportementales calculées depuis l'historique des résultats.

Il utilise une Random Forest entraînée sur l'ensemble des résultats historiques F1.

**Features utilisées :**

- `grid` — position de départ, la feature la plus corrélée au résultat final
- `driver_points` — points au championnat à l'instant T de la course
- `constructor_points` — points de l'écurie, proxy indirect de la qualité de la voiture
- `consistency` — stabilité des performances du pilote sur sa carrière, calculée comme `1 - (écart-type des positions / moyenne des positions)`. Un score proche de 1 signifie que le pilote finit toujours dans les mêmes positions, proche de 0 qu'il est très irrégulier.
- `aggression` — taux de courses terminées sur incident (accident ou collision) par rapport au total de courses disputées. Un score élevé indique un pilote qui prend beaucoup de risques.

**Pourquoi ces deux métriques plutôt que des valeurs fixes ?**

L'ancienne version hardcodait `consistency = 0.8` pour tout le monde, ce qui rendait la feature inutile — le modèle ne pouvait pas distinguer Hamilton de n'importe quel autre pilote sur ce critère. Maintenant chaque pilote a ses propres valeurs calculées depuis son historique réel.

**Méthode `get_driver_behavioral_features(driver_id)`** — permet à l'orchestrateur de récupérer les scores `consistency` et `aggression` d'un pilote pour les injecter dans une prédiction sans ré-entraîner.

**Sortie :** un float entre 1.0 et 20.0 représentant la position estimée.

---

### `models/Races.py` — Modèle central de prédiction

C'est le modèle principal du projet. Il agrège toutes les informations produites par les autres modèles pour prédire la position finale d'un pilote lors d'une course donnée.

Il utilise un Gradient Boosting, l'algorithme le plus adapté aux données tabulaires F1 : il gère bien les interactions entre features (par exemple, un pilote consistant sur un circuit technique) et est robuste aux valeurs aberrantes comme les DNF ou les safety cars.

**Features utilisées (13 au total) :**

- *Grille et course* — `grid`, `round`, `year`
- *Pilote* — `driver_points`, `consistency`, `aggression`, `rolling_avg_position`
- *Constructeur* — `constructor_points`, `reliability_score`
- *Circuit* — `circuit_cluster`, `avg_position_delta`, `avg_dnf_rate`
- *Météo* — `weather_factor`

**`rolling_avg_position`** — moyenne des 5 dernières positions du pilote, calculée avec un décalage temporel strict (`shift(1)`) pour ne jamais utiliser le résultat de la course à prédire. C'est une feature particulièrement importante car elle capte la forme récente du pilote.

**Split temporel** — le modèle est entraîné sur toutes les saisons sauf la dernière, qui sert de jeu de test. Un split aléatoire serait une erreur ici : en F1, utiliser des résultats futurs pour prédire des courses passées est une fuite de données.

**Pourquoi pas un DecisionTree ?**

L'ancienne version utilisait un `DecisionTreeRegressor` sur seulement `['year', 'round']` et prédisait `round * 1.5` — ce qui n'a aucun sens métier. Aucune information sur le pilote ou l'écurie n'était prise en compte.

---

### `models/Strategy.py` — Stratégie de pit stop

Ce modèle prédit le tour optimal pour effectuer un arrêt au stand, pour un pilote donné dans une course donnée.

Il est entraîné sur l'historique complet des pit stops F1, enrichi de contexte sur le pilote, l'écurie et le circuit.

**Features utilisées :**

- `stop` — numéro de l'arrêt (1er, 2ème, 3ème…), la feature la plus déterminante
- `grid` — position de départ, influence la stratégie de sous-cut ou over-cut
- `avg_pit_duration` — durée moyenne des arrêts du pilote en millisecondes, proxy de la fiabilité en pit lane
- `circuit_dnf_rate` — taux de non-finissants sur ce circuit, récupéré depuis `CircuitModel`
- `constructor_reliability` — taux de finitions de l'écurie, calculé depuis l'historique des résultats
- `round` — numéro de manche, capte indirectement la dégradation des pneus selon les règlements de la saison

**Pourquoi pas `raceId` et `stop` comme avant ?**

`raceId` est un identifiant arbitraire — le modèle apprend par cœur les courses passées mais ne peut pas généraliser sur une course future dont l'ID est inconnu. Les nouvelles features sont toutes calculables avant la course.

**Sortie :** un entier entre 1 et 70 représentant le tour recommandé pour l'arrêt.

---

### `models/WeathersConditions.py` — Facteur météo simulé

Ce modèle produit un `weather_factor` entre 0.6 et 1.0 pour chaque course, représentant la stabilité des conditions météo. Il est utilisé par `RaceModel` comme feature et par `F1Predictor` pour moduler le score final.

**Le problème de départ :** le dataset Ergast ne contient aucune donnée météo. L'ancienne version générait des données aléatoires avec `np.random`, ce qui ne reflétait aucune réalité F1.

**La solution — un proxy en trois signaux :**

- **Le mois de la course** — chaque mois a une probabilité de pluie estimée d'après les patterns climatiques historiques des GP (avril et septembre sont les mois les plus pluvieux du calendrier F1).
- **Le circuit** — certains circuits sont structurellement plus pluvieux que d'autres. Spa-Francorchamps reçoit un bonus de +0.25, Silverstone +0.20, Interlagos +0.15, etc.
- **Le `dnf_rate_delta`** — si une course a eu beaucoup plus de non-finissants que la normale pour ce circuit, c'est souvent le signal d'une course sous la pluie ou d'un incident majeur.

Ces trois signaux sont combinés pour construire une target synthétique cohérente, sur laquelle une Random Forest est entraînée.

**Deux méthodes de prédiction :**
- `predict_for_race(race_id)` — retourne le factor d'une course historique déjà indexée lors du train, utilisé par `RaceModel` pendant l'entraînement.
- `predict(dict)` — calcule le factor pour une course future à partir du mois, du circuit et du round, utilisé par le dashboard quand l'utilisateur ajuste le curseur de risque de pluie.