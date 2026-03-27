# 🖥️ Dashboard & UI Management Guide

This guide explains the technical architecture of the **F1 Predictor** frontend, focusing on the interactive Streamlit dashboard and the data transformation layer that powers it.

---

## 📂 1. `display.py`: The User Interface
This file is the primary entry point for the web application. It handles the layout, sidebar inputs, and the rendering of interactive charts and tables.

### Key Layout Modules
* **`render_header()`**: Sets the title and global description of the project.
* **`render_sidebar()`**:
    * Creates the navigation menu for selecting the **Season (Year)** and the **Grand Prix**.
    * It dynamically updates the available races based on the selected year from the database.
* **`render_weather_sidebar()`**:
    * Provides interactive sliders for **Air Temperature**, **Track Temperature**, and **Rain Probability**.
    * These inputs are packaged into a dictionary and passed directly to the `WeatherModel` physics engine.
* **`render_results_table()`**:
    * Displays the core comparison table. It merges historical results with AI predictions (Raw, Initial, and Adjusted).
* **`render_score_chart()` & `render_factors_chart()`**:
    * Uses **Plotly Express** to generate interactive visualizations.
    * The scatter plot helps users visualize the "Car vs. Pilot" performance matrix.
* **`render_winners()`**:
    * Provides a quick visual comparison between the actual race winner and the AI's predicted winner.

---

## 📂 2. `load_to_display.py`: UI Logic & Data Preparation
This file contains the "glue" functions that process API data and Predictor outputs specifically for the frontend requirements.

### Data Processing & Simulation
* **`load_data_and_models()`**:
    * Initializes the entire system. It is decorated with `@st.cache_resource` to ensure that model training happens only once during the session, significantly improving performance.
* **`build_race_dataframe()`**:
    * Performs complex joins (`Results` + `Drivers` + `Constructors` + `Status`) to reconstruct the full starting grid and historical outcome of a specific GP.
* **`run_simulations()`**:
    * Loops through every driver in a selected race. It calls the `F1Predictor` for each and aggregates results into a simulation map.

### Ranking & Accuracy Logic
* **`compute_adjusted_ranks(rank_map, dnf_ids)`**:
    * **Post-Processing**: ML models often predict "ideal" finishes. This function "re-ranks" drivers by removing those who actually retired (DNF).
    * *Example:* If the AI predicts a driver in P3, but the drivers in P1 and P2 retired, this function promotes the driver to a "P1 Adjusted" rank.
* **`build_display_rows(...)`**:
    * Prepares the final dataset for the Streamlit table.
    * **Accuracy Marker (⭐)**: Compares the "Actual Rank" vs. the "Adjusted Predicted Rank". If the difference is $\le 1$, a star is awarded to denote high model precision.
* **`get_predicted_winner()`**:
    * Identifies the driver assigned the `P1` adjusted rank to highlight them as the AI's favorite for the win.

---

## 📈 Dashboard Data Flow Summary

The following diagram illustrates how user inputs are transformed into the final visual dashboard:



---

## 🚥 UI Feature Breakdown

| Feature | Logic Location | Purpose |
| :--- | :--- | :--- |
| **Reverse Y-Axis** | `display: render_score_chart` | In racing, P1 is at the top. The chart reverses the axis so the best scores (lowest numbers) appear highest. |
| **DNF Filtering** | `load_to_display: compute_dnf_ids` | Identifies drivers who crashed or had mechanical failures based on historical status codes. |
| **Team Column Mapping** | `load_to_display: get_team_col` | Handles dataset variations in team naming (e.g., `name_con` vs `name`) automatically. |
| **ML Brut Score** | `display: build_display_rows` | Displays the raw floating-point value from the regressor (e.g., P4.2) before it is discretized into a rank. |

---

### How to Run
To launch the dashboard locally, use the following command in your terminal:
```bash
streamlit run display.py