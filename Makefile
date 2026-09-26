# TailSafe developer commands. Run `make help` for a summary.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
STAMP  := $(VENV)/.installed
PORT   ?= 8000

.DEFAULT_GOAL := help
.PHONY: help install test test-slow test-all cov lint format typecheck check dev api schema demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(STAMP): pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -q --upgrade pip
	$(BIN)/python -m pip install -q -e ".[dev]"
	@touch $(STAMP)

install: $(STAMP) ## Create .venv and install tailsafe with dev dependencies

test: $(STAMP) ## Run the fast test suite
	$(BIN)/pytest

test-slow: $(STAMP) ## Run only the slow tests (performance targets, large Monte Carlo)
	$(BIN)/pytest -m slow

test-all: $(STAMP) ## Run every test
	$(BIN)/pytest -m ""

cov: $(STAMP) ## Fast test suite with a coverage report
	$(BIN)/pytest --cov --cov-report=term-missing

lint: $(STAMP) ## Lint and check formatting
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: $(STAMP) ## Auto-format and apply safe lint fixes
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: $(STAMP) ## Static type checking
	$(BIN)/mypy

check: lint typecheck test ## Everything CI runs

dev: api ## Start the development server(s)

api: $(STAMP) ## Start the FastAPI backend with auto-reload on http://localhost:$(PORT)
	$(BIN)/uvicorn tailsafe.api.app:app --reload --port $(PORT)

schema: $(STAMP) ## Regenerate schemas/*.json from the Pydantic models
	$(BIN)/tailsafe schema export

demo: $(STAMP) ## Generate and render the 40-storey cruciform demo building
	$(BIN)/tailsafe building generate cruciform --storeys 40 --out out/cruciform_40.json
	$(BIN)/tailsafe building render out/cruciform_40.json --out out/cruciform_40.png

clean: ## Remove caches and build outputs (keeps .venv)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist out
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
