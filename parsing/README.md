# 🛠️ Data Ingestion Pipeline

This directory contains the logic required to load, validate, and clean historical Formula 1 data (CSV format).

---

## 📂 1. `load.py`: File Loading
This file is responsible for the raw reading of data from the storage disk.

### Functions
* **`_load_single_file(key, path)`**:
    * **Role**: Attempts to read a specific CSV file using Pandas.
    * **Logic**: Handles non-existent paths and replaces Ergast-specific missing data markers (`\N` or `\\N`) with actual `NaN` values.
* **`load_data()`**:
    * **Role**: Orchestrates the loading of all tables defined in `FILES_MAP`.
    * **Logic**: Dynamically resolves the path to the `stats/` folder and returns a dictionary containing the loaded DataFrames.

---

## 🏗️ 2. `parser.py`: Audit & Validation
This file ensures data integrity before it is sent to the Machine Learning models. It uses a strict **SCHEMA** to validate every table.

### Key Variables
* **`SCHEMA`**: Defines the expected columns, **critical** (mandatory) columns, and data types (e.g., `Int64`, `Float64`) for each table.
* **`BLOCKING_TABLES`**: A list of vital tables (`races`, `results`, `drivers`, `constructors`). If any of these are corrupted or missing, the program safely stops.
* **`NAN_THRESHOLD`**: Set at `0.30` (30%). If a non-critical column has more than 30% missing data, it is flagged or removed.

### Processing Functions
* **`cast_dataframe_types(key, df)`**:
    * Forces column typing according to the schema to prevent calculation errors (e.g., converting an ID column read as text into an integer).
* **`_clean_single_table(key, df)`**:
    * Removes non-critical columns that exceed the `NAN_THRESHOLD`, while preserving vital data.

### Audit Functions
* **`_classify_single_table(key, df)`**:
    * Analyzes a table and assigns a status: `ok`, `suspect` (missing critical columns), `vide` (empty), or `absent`.
* **`build_health_report(dataframes, status)`**:
    * Generates a detailed dictionary regarding data health (row counts, missing columns, fill rates).
* **`print_health_report(report)`**:
    * Displays a comprehensive visual report in the terminal for the user.

### Main Entry Point
* **`parse_and_audit(dataframes)`**:
    * The single entry point that chains: type conversion, classification, health report display, user confirmation prompt (if anomalies exist), and final filtering.

---

## 🚦 Table Statuses

| Status | Action |
| :--- | :--- |
| **OK (✓)** | The table is complete and ready for use. |
| **SUSPECT (⚠)** | Missing critical columns or too many `NaN` values. Manual user confirmation is required. |
| **ABSENT/EMPTY (✗)** | The table is ignored. If listed in `BLOCKING_TABLES`, the system triggers a safety shutdown. |