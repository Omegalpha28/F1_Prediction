# F1 Race Prediction System / F1_Prediction

A modular machine learning pipeline for Formula 1 race outcome prediction. The system models driver behavior, constructor performance, circuit characteristics, and weather conditions to simulate race results through an interactive dashboard.

---

## Table of Contents

1. [Installation](#installation)
2. [Running the Application](#running-the-application)
3. [Utility Commands](#utility-commands)
4. [Tech Stack & Dependencies](#tech-stack--dependencies)
5. [Architecture Overview](#architecture-overview)
6. [Module Reference](#module-reference)
   - [Data Layer](#data-layer)
   - [Machine Learning Models](#machine-learning-models)
   - [Orchestration & API](#orchestration--api)

---

## Installation

This project manages its own isolated virtual environment via a `Makefile`, keeping your global Python installation untouched.

**Prerequisites:** `make` and Python 3 must be installed on your machine.

```bash
make
```

This single command will:

- Create a local virtual environment (`.venv`)
- Install all required dependencies
- Generate a standalone executable named `F1_Dashboard`

---

## Running the Application

Once the build is complete, launch the interactive dashboard using either method:

```bash
# Via the generated executable
./F1_Dashboard

# Via Make
make run
```

---

## Utility Commands

| Command       | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `make`        | Full build: creates `.venv`, installs deps, compiles executable    |
| `make run`    | Launches the application                                           |
| `make test`   | Verifies the virtual environment and scientific library imports     |
| `make clean`  | Removes temporary files and `__pycache__` directories              |
| `make fclean` | Deep clean: removes the executable and the virtual environment     |
| `make re`     | Full reset — equivalent to `fclean` followed by `make`             |

---

## Tech Stack & Dependencies

| Category               | Technology / Library                          |
|------------------------|-----------------------------------------------|
| Language               | Python 3                                      |
| Data Manipulation      | pandas                                        |
| Machine Learning       | scikit-learn (RandomForest, GradientBoosting, KMeans, StandardScaler) |
| Data Source            | F1 Dataset (CSV files)                 |
| Build System           | GNU Make                                      |
| Runtime Isolation      | Python `venv`                                 |
| Interface              | Streamlit                                     |

---

## Architecture Overview

The pipeline follows a strict layered architecture. Data flows in one direction: from raw CSV files through cleaning and auditing, into domain-specific ML models, and finally into the race orchestrator that produces the final prediction.

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│                                                             │
│   stats/ (CSV files)                                        │
│        │                                                    │
│        ▼                                                    │
│   load.py          ──►  parser.py                           │
│   (Filesystem I/O)       (Schema validation & cleaning)     │
│        │                                                    │
│        ▼                                                    │
│   api.py (F1API)   ──  Single source of truth for all data  │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                    ML MODEL LAYER                           │
│                                                             │
│   Ml_Predictions.py   (Abstract base — Scaler & medians)    │
│         │                                                   │
│         ├──► Circuits.py      (KMeans — circuit profiles)   │
│         ├──► Constructor.py   (RandomForest — car perf.)    │
│         ├──► Pilots.py        (RandomForest — driver style) │
│         ├──► WeathersConditions.py  (Physics engine)        │
│         ├──► Strategy.py      (RandomForest — pit stops)    │
│         └──► Races.py         (GradientBoosting — final)    │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                      ORCHESTRATOR                           │
│                                                             │
│   prediction.py  ──  Aggregates all model outputs           │
│        │             Applies weather & overtaking modifiers │
│        ▼                                                    │
│   Dashboard  ──  Final race simulation & report             │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### Data Layer

#### `load.py` — Filesystem Reader

Handles all CSV file loading using relative path resolution (`os.path.dirname`) to ensure the project runs correctly regardless of where it is launched from. A `FILES_MAP` dictionary restricts loading to explicitly expected files only, ignoring unrelated data in the directory. A `try/except` block ensures the program halts safely if any required file is missing.

> **Role in pipeline:** Entry point. If loading fails, execution stops before any model is touched.

---

#### `parser.py` — Schema Validator & Data Auditor

Enforces a strict `SCHEMA` dictionary defining expected columns, critical columns, and data types for all 14 F1 tables. Tables are categorized as `OK`, `missing`, or `suspect`. Non-critical columns with more than 30% missing values (`\N`) are automatically dropped. A `BLOCKING_TABLES` rule prevents execution if core tables such as `results` or `races` are corrupted.

> **Role in pipeline:** Data integrity gate. Guarantees that ML models never encounter unexpected nulls or type errors. Outputs a formatted Health Report to the terminal before the simulation proceeds.

---

#### `api.py` — F1API (Data Access Interface)

Implements the **Facade** pattern combined with a **Data Access Object (DAO)** structure. Loads the full dataset into memory once via `load_and_audit()` and exposes domain-specific query methods that delegate to specialized sub-modules in the `getters/` folder (organized by drivers, circuits, and constructors).

> **Role in pipeline:** Central distribution hub. All ML models and the orchestrator query the API (e.g., `api.get_training_matrix()`) rather than reading files directly. Ensures a single copy of the dataset resides in memory at all times.

---

### Machine Learning Models

#### `Ml_Predictions.py` — Abstract Base Class

Defines the shared infrastructure for all ML models: a global `StandardScaler` (normalizing features to mean 0, variance 1) and a median memory system that stores training-phase baselines to prevent **data leakage** during inference.

> **Role in pipeline:** Foundation. All models inherit from this class, guaranteeing uniform data transformation across the entire pipeline.

---

#### `Circuits.py` — CircuitModel

Uses **unsupervised learning (KMeans clustering)** to group circuits into 4 physical profiles: street, high-speed, balanced, and technical. Dynamically computes two circuit-level metrics: `overtaking_rate` and `avg_dnf_rate` (danger level).

> **Role in pipeline:** Context provider. Downstream models use circuit profiles to adapt predictions — for example, an aggressive driver incurs a higher penalty on a dangerous street circuit.

---

#### `Constructor.py` — ConstructorModel

Applies a **RandomForestRegressor** with rolling time windows to capture team development dynamics. Computes `development_trend` by subtracting the long-term points average from the short-term one, detecting whether a constructor is improving or declining mid-season.

> **Role in pipeline:** Quantifies car potential. Feeds the race model a baseline mechanical competitiveness score for each constructor.

---

#### `Pilots.py` — PilotModel

A **RandomForestRegressor** focused on behavioral profiling. Computes two driver-specific indices: `pos_std` (position standard deviation — consistency) and `aggression` (crash frequency).

> **Role in pipeline:** Generates the driver performance index. Combined with the CircuitModel, anticipates whether a driver's style suits the track layout and race conditions.

---

#### `WeathersConditions.py` — WeatherModel

A hybrid model using two strategies:

- **Historical races (proxy deduction):** Infers past weather from DNF rate spikes relative to circuit baseline.
- **Live simulation (physics engine):** Applies three cumulative penalty factors:

| Factor          | Formula                                                    | Max Penalty |
|-----------------|------------------------------------------------------------|-------------|
| Humidity        | `P_rain = P(rain) × 0.30`                                  | −0.30       |
| Track temp.     | `P_grip = min((T_track − 35) / 40, 1.0) × 0.15`           | −0.15       |
| Cloud cover     | `P_clouds = (1 − sigmoid(0.5 × (ΔT − 5))) × 0.10`         | −0.10       |

The resulting **Weather Score** ranges from `1.0` (ideal) to `0.5` (extreme chaos).

> **Role in pipeline:** Global chaos modifier. A degraded score alters predicted finishing order, caps the overtaking multiplier (below 0.90 threshold), and scales up DNF probabilities.

---

#### `Strategy.py` — StrategyModel

A **RandomForestRegressor** that cross-references grid position, constructor reliability, and circuit danger rate to predict the `optimal_pit_lap` for each driver.

> **Role in pipeline:** Tactical enrichment. The optimal pit stop lap is surfaced directly in the dashboard's simulation report.

---

#### `Races.py` — RaceModel

The final prediction model, built on **GradientBoostingRegressor**. Ingests only contextual features (aggressiveness, weather score, circuit cluster) and deliberately excludes temporal identifiers (year, race ID) to force the algorithm to generalize from situations rather than memorize historical outcomes.

> **Role in pipeline:** Final judge. Synthesizes all upstream model outputs into a raw finishing position prediction, before the overtaking difficulty modifier is applied by the orchestrator.

---

### Orchestration & API

#### `prediction.py` — Orchestrator

Coordinates the full pipeline execution: queries the F1API, feeds data through each model in sequence, applies the weather and overtaking modifiers to the raw predictions, and assembles the final race simulation report delivered to the dashboard.

> **Role in pipeline:** End-to-end coordinator. The dashboard interacts exclusively with this module.