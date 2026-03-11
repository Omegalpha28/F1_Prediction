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


