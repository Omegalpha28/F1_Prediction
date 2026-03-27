.PHONY: all venv install run dashboard build test clean fclean re

VENV_DIR := .venv
PYTHON_REQUIRED := 3.13
PYTHON3_13 := $(shell command -v python3.13 2>/dev/null || true)
PYTHON3 := $(shell command -v python3 2>/dev/null || true)
SYSTEM_PYTHON := $(or $(PYTHON3_13),$(PYTHON3))
VENV_PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
STREAMLIT := $(VENV_DIR)/bin/streamlit

REQUIREMENTS := requirement.txt
SCRIPT := main.py
EXECUTABLE := F1_Dashboard

all: install build

venv:
	@if [ -z "$(SYSTEM_PYTHON)" ]; then \
		echo "[ERROR] Aucun Python trouvé. Installe Python $(PYTHON_REQUIRED) ou pyenv."; exit 1; \
	fi
	@test -d $(VENV_DIR) || $(SYSTEM_PYTHON) -m venv $(VENV_DIR)

install: venv
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r $(REQUIREMENTS)

run:
	@$(VENV_PYTHON) $(SCRIPT)


build: $(SCRIPT)
	@echo "#!$(VENV_PYTHON)" > $(EXECUTABLE)
	@cat $(SCRIPT) >> $(EXECUTABLE)
	@chmod +x $(EXECUTABLE)
	@echo "✅ Binary created : ./$(EXECUTABLE)"

test:
	@$(VENV_PYTHON) -c "import pandas, numpy, scipy, sklearn, matplotlib, seaborn; print('✅ Environnement prêt !')"

clean:
	@rm -f $(EXECUTABLE)
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -name '*.pyc' -delete

fclean: clean
	@rm -rf $(VENV_DIR)

re: fclean all