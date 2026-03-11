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


