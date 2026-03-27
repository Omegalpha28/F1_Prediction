# 🧠 Predictive Architecture: The Encapsulation Approach

This project is built on a **hierarchical modular architecture**. Instead of using a single "catch-all" model, we have fragmented the reality of a Grand Prix into **specialized child models**.

### The Philosophy
The core idea is that the final model (**RaceModel**) cannot understand the complexity of a race unless it receives pre-processed information from "domain experts."
1. **Child Models** (`Circuit`, `Constructor`, `Pilot`, `Weather`, `Strategy`) analyze specific segments and extract "business scores."
2. **Encapsulation** feeds the parent model with these scores. 
   * *Example:* The race model doesn't just see a circuit "ID"; it receives an "Overtaking Rate" calculated by the expert `CircuitModel`.
3. **Tactical Synergy:** The `RaceModel` acts as a conductor, synthesizing these expertises to predict the final result.

---

## 🏗️ 1. `Ml_Prediction.py` (The Mother Class)
The common foundation for all models.
* **`preprocess_features`**: Cleans data, converts columns to numeric, and handles missing values via **median memory** (prevents errors if data is missing during a real-time prediction).
* **`save_model` / `load_model`**: Persists the models' intelligence (weights, scalers, medians) to the disk so they don't need to be re-trained every time.

---

## 🏁 2. `CircuitModel.py`
The environmental expert. It segments the circuits of the F1 calendar.
* **`_compute_overtaking_rate`**: Mathematically calculates the difficulty of overtaking on a track by analyzing the historical delta between grid position and finish.
* **`_fit_model` (KMeans)**: Uses unsupervised learning to classify circuits into 4 clusters: Street, High Speed, Balanced, and Technical.
* **`get_circuit_features`**: Provides the parent model with the complete circuit profile.

---

## 🏎️ 3. `ConstructorModel.py`
The mechanical expert. It evaluates team performance.
* **`_compute_reliability_score`**: Calculates the finishing rate (reliability) for each team, which is vital for predicting DNFs (Did Not Finish).
* **`_enrich_constructor_form`**: Analyzes "momentum" by comparing recent points to historical averages (`development_trend`).
* **`predict`**: Returns the expected position of the car based solely on its mechanical strength.

---

## 👤 4. `PilotModel.py`
The human expert. It profiles driver behavior.
* **`_compute_behavioral_features`**: Analyzes stability (`pos_std`) and aggressiveness (`aggression`) based on past incidents.
* **`_enrich_rolling_avg_position`**: Calculates the driver's current form over the last 5 races.
* **`predict`**: Evaluates the position a driver is capable of achieving, independent of their car.

---

## ⛈️ 5. `WeatherModel.py`
The physics engine. It simulates the impact of climatic conditions.
* **`train`**: Analyzes history to link DNF rates to past weather conditions.
* **`predict`**: Applies mathematical formulas (Rain penalties, Track temperature sigmoid) to generate a `weather_factor`.
  * *Note:* This factor directly impacts grip and overtaking difficulty.

---

## 🚦 6. `RaceModel.py` (The Parent / Orchestrator)
The final brain performing the synthesis.
* **`train`**: This is where encapsulation comes to life. It calls each child model to enrich its training matrix.
* **`pos_gain`**: **Key Prediction Strategy**. The model does not predict a raw position (1 to 20), but rather the **number of places gained or lost** (`grid - positionOrder`). This prevents unrealistic predictions (like a driver starting P1 but predicted P10 for no reason).
* **`_apply_weather_modifier`**: A "Blended Learning" function. It adjusts the ML prediction based on weather: the worse the weather, the more the starting position (`grid`) is weighted, as overtaking becomes mathematically riskier.

---

## 🛠️ 7. `StrategyModel.py`
The tactical expert. It predicts pit stops.
* **`_add_avg_pit_duration`**: Analyzes the speed of a driver/team's pit crew.
* **`predict`**: Cross-references the grid, reliability, and circuit danger to estimate the optimal lap for the first pit stop.

---

### ML Data Flow Summary

1. **Input**: Circuit, Driver, Team, Weather sliders.
2. **Children**: Calculate scores (Aggression, Reliability, Cluster, Weather Factor).
3. **Parent**: Receives these scores + the Grid position. Predicts the `pos_gain`.
4. **Output**: Final adjusted position based on today's real overtaking difficulty.