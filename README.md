# F1_Detector

A realistic F1 race prediction model built with Python and machine learning.

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
- `main.py` loads all CSV files on startup and verifies that no table is empty before launching the dashboard.

## Note
To ensure the project uses environment Python and packages (instead of system ones), activate the venv or run with `./.venv/bin/python`.

---

## API — `api.py`

The `F1API` class is the single access point to all historical F1 data stored in `stats/*.csv`. Every ML model and the predictor use it exclusively — no model reads CSV files directly.

It also exposes two higher-level helpers used during training:
- `get_training_matrix()` — builds an enriched DataFrame by merging results with races, driver standings, and constructor standings. This is the base dataset used by `PilotModel`, `ConstructorModel`, and `RaceModel`.
- `get_driver_history_before(driver_id, year)` — returns a driver's average historical finishing position, used to detect rookies in `F1Predictor`.
- `get_constructor_history_before(constructor_id, year)` — same for constructors.

### Getter methods

- **Global** — `get_table(table_name)`, `get_table_by(table_name, **filters)`
- **Drivers** — `get_all_drivers()`, `get_driver_by_id()`, `get_driver_by_ref()`, `get_drivers_by_nationality()`, `get_drivers_by_name()`
- **Circuits** — `get_all_circuits()`, `get_circuit_by_id()`, `get_circuit_by_ref()`, `get_circuits_by_country()`
- **Constructors** — `get_all_constructors()`, `get_constructor_by_id()`, `get_constructor_by_ref()`, `get_constructors_by_nationality()`
- **Races** — `get_all_races()`, `get_race_by_id()`, `get_races_by_year()`, `get_races_by_circuit()`, `get_races_by_name()`
- **Results** — `get_all_results()`, `get_result_by_id()`, `get_results_by_race()`, `get_results_by_driver()`, `get_results_by_constructor()`, `get_results_by_status()`
- **Sprint Results** — `get_all_sprint_results()`, `get_sprint_results_by_race()`, `get_sprint_results_by_driver()`
- **Constructor Results** — `get_constructor_results_by_race()`, `get_constructor_results_by_constructor()`
- **Standings** — `get_all_driver_standings()`, `get_driver_standings_by_race()`, `get_driver_standings_by_driver()`, `get_all_constructor_standings()`, `get_constructor_standings_by_race()`, `get_constructor_standings_by_constructor()`
- **Lap / Pit / Qualifying** — `get_all_lap_times()`, `get_lap_times_by_race()`, `get_lap_times_by_driver()`, `get_all_pit_stops()`, `get_pit_stops_by_race()`, `get_pit_stops_by_driver()`, `get_all_qualifying()`, `get_qualifying_by_race()`, `get_qualifying_by_driver()`, `get_qualifying_by_constructor()`
- **Seasons / Status** — `get_all_seasons()`, `get_season_by_year()`, `get_all_status()`, `get_status_by_id()`

### Example
```python
from api import F1API
api = F1API()
drivers = api.get_all_drivers()
hamilton = api.get_driver_by_ref('hamilton')
results_race_1 = api.get_results_by_race(1)
```

---

## ML Architecture

The prediction system is organized as a hierarchy of ML classes. Every model inherits from a shared abstract base and is specialized for one specific task.

```
Ml_Prediction  (abstract base)
    ├── CircuitModel       →  circuit clustering + scoring
    ├── ConstructorModel   →  constructor performance
    ├── PilotModel         →  driver performance
    ├── WeatherModel       →  weather factor simulation
    ├── StrategyModel      →  optimal pit stop lap
    └── RaceModel          →  final race position  ← main model
              ↑
    F1Predictor (orchestrator) — trains all models in order, assembles features, runs simulations
```

---

### `models/Ml_Prediction.py` — Abstract base class

The base class inherited by all ML models in the project. It centralizes shared logic to avoid code duplication across child models.

**What it provides to all children:**

- **`preprocess_features(df, features)`** — Cleans a DataFrame before training or prediction. Handles missing values (`\N` from the Ergast format and `NaN`), validates that requested columns exist, and casts everything to `float`. Order matters: `\N` strings are replaced with `NaN` first, then `fillna` with the column median, then cast.
- **`get_feature_names()`** — Returns the list of features consumed by the model. Used by `F1Predictor` to know which columns each model expects.
- **`save_model(path)` / `load_model(path)`** — Persistence via `joblib`. Saves and reloads the trained model, scaler, and feature list in a single file. Shared by all children — none should override these methods.

**What each child must implement:**

- **`train(api)`** — Model training. Receives the `F1API` instance so it can fetch whatever data it needs internally.
- **`predict(input_data)`** — Generates a prediction from an input. The type of `input_data` depends on the child model (dict, DataFrame, etc.).

**Why `ABC` instead of bare `raise NotImplementedError`?**
With `ABC`, Python refuses to instantiate `Ml_Prediction` directly and raises an error immediately if a child forgets to implement `train` or `predict`. The old approach silently allowed instantiation until the method was actually called.

---

### `models/Circuits.py` — Circuit classification

This model answers two questions: **what type of circuit is this?** and **how easy is it to overtake there?**

It uses a KMeans clustering algorithm to automatically group circuits into 4 families based on historical and geographic features. These labels and scores are then consumed by `RaceModel` and `StrategyModel` as additional context.

**What it computes during training:**

- **Circuit type** — by combining the circuit's GPS coordinates (latitude, longitude, altitude) with two metrics derived from historical results. Each circuit is assigned a label: `street_circuit`, `high_speed`, `balanced`, or `technical`.
- **`avg_position_delta`** — the average number of positions gained or lost by drivers on this circuit (difference between starting grid and finishing position). A high value means overtaking is frequent.
- **`avg_dnf_rate`** — the historical rate of drivers who did not finish on this circuit. A proxy for danger level and mechanical demands.

**What other models consume:**

Via `get_circuit_features(circuit_id)`, the orchestrator retrieves a complete feature dictionary for each circuit and injects it into `RaceModel` and `StrategyModel`.

---

### `models/Constructor.py` — Constructor performance

This model predicts the expected race finishing position for a given constructor, based on its championship statistics and historical mechanical reliability.

It uses a Random Forest trained on the full results history merged with constructor standings.

**Features used:**

- `constructor_points` — championship points at the time of the race
- `constructor_wins` — victories in the current season
- `constructor_champ_pos` — current position in the constructors' championship
- `reliability_score` — rate of races finished by the constructor, computed automatically from results (`statusId == 1` / total races). Each team gets its own real score — Red Bull and Mercedes will naturally score higher than less reliable teams.

**Output:** a float between 1.0 and 20.0. Used by `F1Predictor` as a component of the final score in `simulate_race()`.

---

### `models/Pilots.py` — Driver performance

This model predicts the expected race finishing position for a given driver, based on championship statistics and two behavioral metrics derived from historical results.

It uses a Random Forest trained on the full F1 results history.

**Features used:**

- `grid` — starting position, the feature most correlated with the final result
- `driver_points` — championship points at the time of the race
- `constructor_points` — constructor points, an indirect proxy for car quality
- `consistency` — stability of the driver's performances, computed as `1 - (std(positions) / mean(positions))`. A score close to 1 means the driver always finishes in similar positions; close to 0 means highly erratic results.
- `aggression` — rate of races ended in an incident (accident or collision) out of total races started. A high score indicates a risk-taking driver.

**Why compute these metrics instead of using fixed values?**
The old version hardcoded `consistency = 0.8` for everyone, making the feature useless — the model couldn't distinguish Hamilton from any other driver on this criterion. Now each driver has their own values computed from their real career history.

**`get_driver_behavioral_features(driver_id)`** — allows the orchestrator to retrieve a driver's `consistency` and `aggression` scores and inject them into a prediction without retraining.

**Output:** a float between 1.0 and 20.0.

---

### `models/Races.py` — Main race prediction model

This is the central model of the project. It aggregates all features produced by the other models to predict the final finishing position of a driver in a given race.

It uses Gradient Boosting, the best-suited algorithm for tabular F1 data: it handles non-linear relationships and feature interactions well (e.g. a consistent driver on a technical circuit), and is robust to outliers like DNFs and safety cars.

**Features used (13 total):**

| Category | Features |
|---|---|
| Grid & race | `grid`, `round`, `year` |
| Driver | `driver_points`, `consistency`, `aggression`, `rolling_avg_position` |
| Constructor | `constructor_points`, `reliability_score` |
| Circuit | `circuit_cluster`, `avg_position_delta`, `avg_dnf_rate` |
| Weather | `weather_factor` |

**`rolling_avg_position`** — the driver's average finishing position over their last 5 races, computed with a strict `shift(1)` to never include the current race result. This captures recent form and is one of the most predictive features.

**Temporal split** — the model is trained on all seasons except the most recent one, which is used as the test set. A random split would be wrong here: using future race results to predict past races is a data leak.

**Why not a DecisionTree?**
The old version used a `DecisionTreeRegressor` on only `['year', 'round']` and effectively predicted `round * 1.5` — which has no business meaning and ignores driver and constructor information entirely.

---

### `models/Strategy.py` — Pit stop strategy

This model predicts the optimal lap to make a pit stop, for a given driver in a given race.

It is trained on the complete F1 pit stop history, enriched with context about the driver, constructor, and circuit.

**Features used:**

- `stop` — pit stop number (1st, 2nd, 3rd…), the most deterministic feature
- `grid` — starting position, influences undercut and overcut timing
- `avg_pit_duration` — the driver's average pit stop duration in seconds, a proxy for pit lane reliability
- `circuit_dnf_rate` — DNF rate for this circuit, pulled from `CircuitModel`
- `constructor_reliability` — constructor finish rate, computed from historical results
- `round` — race number in the season, indirectly captures tyre degradation under different regulations

**Why not `raceId` and `stop` as before?**
`raceId` is an arbitrary identifier — the model memorizes past races but cannot generalize to a future race with an unknown ID. All new features are computable before the race starts.

**Output:** an integer between 1 and 70 representing the recommended pit stop lap.

---

### `models/WeathersConditions.py` — Simulated weather factor

This model produces a `weather_factor` between 0.6 and 1.0 for each race, representing the stability of weather conditions. It is used by `RaceModel` as a training feature and by `F1Predictor` to modulate the final predicted position.

**The problem:** the Ergast dataset contains no real weather data. The old version generated purely random values with `np.random`, which reflected no F1 reality.

**The solution — a three-signal proxy:**

- **Race month** — each month has an estimated rain probability based on historical F1 GP climate patterns (April and September are the wettest months on the calendar).
- **Circuit identity** — some circuits are structurally wetter than others. Spa-Francorchamps gets a +0.25 bonus, Silverstone +0.20, Interlagos +0.15, etc.
- **`dnf_rate_delta`** — if a race had significantly more non-finishers than the historical average for that circuit, it is likely a signal of rain or a major incident.

These three signals are combined to build a coherent synthetic target, on which a Random Forest is trained.

**Two prediction methods:**
- `predict_for_race(race_id)` — returns the weather factor for a known historical race, indexed during training. Used by `RaceModel` while building its training dataset.
- `predict(dict)` — computes the factor for a future race from month, circuit, and round. Used by the dashboard when the user adjusts the rain probability slider.

---

### `prediction.py` — Orchestrator (`F1Predictor`)

`F1Predictor` is the central coordinator of the entire ML pipeline. It owns all model instances and is responsible for training them in the correct dependency order and assembling their outputs into a final race simulation.

**`train_all()`** — trains all six models in the correct order:
1. `CircuitModel` — must run first, its outputs are consumed by `StrategyModel` and `RaceModel`
2. `WeatherModel` — must run before `RaceModel`
3. `ConstructorModel` and `PilotModel` — independent, can run in any order
4. `StrategyModel` — depends on `CircuitModel`
5. `RaceModel` — depends on all previous models

Each model is trained in an isolated try/except block — a failure in one model does not prevent the others from training. A summary of trained vs. failed models is logged at the end.

**`simulate_race(driver_id, circuit_id, grid_pos, weather_conditions, current_year)`** — simulates the final position for a single driver in a given race. It assembles features from all trained models, computes a base score weighted between driver performance and constructor potential, applies circuit overtaking difficulty and weather impact, and returns a prediction dict with `expected_position_value`, `expected_position_str`, and a `factors` breakdown.

---

### `dashboard.py` — Streamlit interface

The interactive web dashboard. It is launched automatically by `main.py` via `subprocess` and runs as a standalone Streamlit app.

**What it does:**
- Loads data and trains all models once on startup via `@st.cache_resource` — subsequent interactions reuse the cached models without retraining.
- Lets the user select a year and a Grand Prix from the sidebar, then adjust weather conditions (air temperature, track temperature, rain probability) via sliders.
- On simulation launch, runs `simulate_race()` for every driver in the historical race and displays a results table comparing real finishing positions against ML predictions.
- Shows two charts: a bar chart of ML scores per driver, and a scatter plot of driver performance vs. constructor power for each team.
- Highlights the real race winner vs. the predicted winner.

---

### `main.py` — Entry point

Loads all CSV files from `stats/`, verifies that no table is empty, logs a summary of available vs. missing tables, then launches the Streamlit dashboard via `subprocess`.

Running `python main.py` (or `make run`) is the only command needed to start the full application.