import logging
import os
import subprocess
import sys
from parsing.load   import load_data
from parsing.parser import parse_and_audit

logger = logging.getLogger(__name__)

def load_and_audit() -> dict:
    dataframes = load_data()
    return parse_and_audit(dataframes)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        usable = load_and_audit()
        print(f"Tables prêtes pour l'entraînement : {list(usable.keys())}")
    except RuntimeError as exc:
        print(f"\nERREUR BLOQUANTE : {exc}")
        sys.exit(1)
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard/display.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])