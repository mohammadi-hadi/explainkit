PY ?= python3
VENV = .venv
BIN = $(VENV)/bin

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests
	$(BIN)/mypy src

demo:
	$(BIN)/explainkit demo --out results

readme: demo
	$(BIN)/explainkit inject-readme README.md --results results

.PHONY: setup test lint demo readme
