# ==========================================
# Makefile
# ==========================================
# This Makefile assumes:
#   - simulation.py: runs the full simulation and writes CSV tables under results/tables/
#   - plot.py      : reads tables and writes figures under results/figures/
#
# Core targets:
#   make all       : Run complete simulation pipeline and generate all outputs
#   make simulate  : Run simulations and save tables
#   make figures   : Create all figures
#   make clean     : Remove generated files
#   make test      : Run tests
#
# ------------------------------------------

# ---- Basic variables ----
PY          ?= python3
SRC_DIR     ?= src
RESULTS_DIR ?= results
TABLES_DIR  ?= $(RESULTS_DIR)/tables
FIGURES_DIR ?= $(RESULTS_DIR)/figures
SIM_SCRIPT  := $(SRC_DIR)/simulation.py
PLOT_SCRIPT := $(SRC_DIR)/plot.py

.PHONY: all simulate figures clean test

# ------------------------------------------
# Run complete pipeline: simulate -> figures
# ------------------------------------------
all: simulate figures
	@echo "[all] Pipeline complete."

# ------------------------------------------
# Simulations (CSV under $(TABLES_DIR))
# ------------------------------------------
simulate: $(TABLES_DIR)
	@echo "[simulate] Running simulations via $(SIM_SCRIPT)"
	@$(PY) $(SIM_SCRIPT)
	@echo "[simulate] Tables are in $(TABLES_DIR)"

$(TABLES_DIR):
	@mkdir -p $(TABLES_DIR)

# ------------------------------------------
# Figures
# ------------------------------------------
figures: $(FIGURES_DIR)
	@echo "[figures] Creating figures via $(PLOT_SCRIPT)"
	@$(PY) $(PLOT_SCRIPT)
	@echo "[figures] Complete! Figures are in $(FIGURES_DIR)"

$(FIGURES_DIR):
	@mkdir -p $(FIGURES_DIR)

# ------------------------------------------
# Clean
# ------------------------------------------
clean:
	@echo "[clean] Removing generated results under $(RESULTS_DIR)"
	@rm -rf $(RESULTS_DIR)
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@echo "[clean] Done."

# ------------------------------------------
# Tests
# ------------------------------------------
test:
	@echo "[test] Running pytest..."
	@$(PY) -m pytest -q
	@echo "[test] Test complete!"