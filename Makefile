PYTHON ?= python
PIP ?= $(PYTHON) -m pip
LC ?= lc

.PHONY: install dev fmt lint type test run show report clean

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e '.[dev]'

fmt:
	ruff format .
	isort .
	black .

lint:
	ruff check .

type:
	mypy src

test:
	pytest

run:
	$(LC) run

show:
	$(LC) show

report:
	$(PYTHON) -m llm_crypto_fund --config config.yaml show

clean:
	rm -rf .pytest_cache .mypy_cache build dist
