# TailSafe developer commands. Run `make help` for a summary.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
STAMP  := $(VENV)/.installed
PORT   ?= 8000
WEB_STAMP := web/node_modules/.installed

.DEFAULT_GOAL := help
.PHONY: help install test test-slow test-all cov lint format typecheck check dev api schema demo stress-demo pitch-demo clean web web-install web-dev web-check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(STAMP): pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -q --upgrade pip
	$(BIN)/python -m pip install -q -e ".[dev,surrogate]"
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

dev: $(STAMP) $(WEB_STAMP) ## Start the API (:8000) and the web UI with hot reload (:5173)
	$(MAKE) -j2 api web-dev

api: $(STAMP) ## Start the FastAPI backend with auto-reload on http://localhost:$(PORT)
	$(BIN)/uvicorn tailsafe.api.app:app --reload --port $(PORT)

$(WEB_STAMP): web/package-lock.json
	cd web && npm ci --no-audit --no-fund
	@touch $(WEB_STAMP)

web-install: $(WEB_STAMP) ## Install the web UI's npm dependencies

web-dev: $(WEB_STAMP) ## Start the Vite dev server (proxies /api to the backend)
	cd web && npm run dev

web: $(WEB_STAMP) ## Build the web UI into web/dist (then `make api` serves it on :8000)
	cd web && npm run build

web-check: $(WEB_STAMP) ## Type-check and unit-test the web UI (what CI runs for web/)
	cd web && npm run typecheck && npm test

schema: $(STAMP) ## Regenerate schemas/*.json from the Pydantic models
	$(BIN)/tailsafe schema export

demo: $(STAMP) ## Generate and render the 40-storey cruciform demo building
	$(BIN)/tailsafe building generate cruciform --storeys 40 --out out/cruciform_40.json
	$(BIN)/tailsafe building render out/cruciform_40.json --out out/cruciform_40.png

clean: ## Remove caches and build outputs (keeps .venv)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist out
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +

stress-demo: $(STAMP) ## Monte Carlo stress test of the pitch scenario (1,000 runs)
	$(BIN)/tailsafe stress run cruciform --spec demo --runs 1000 --out out/stress-demo

pitch-demo: $(STAMP) ## The pitch end to end (stress, bottlenecks, plan, replay, briefing; ~4 min)
	$(BIN)/tailsafe demo --out out/pitch
