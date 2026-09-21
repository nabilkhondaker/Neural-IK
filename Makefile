.PHONY: install test lint format experiment clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -q

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

experiment:
	python scripts/run_experiment.py --name baseline --samples 5000 --epochs 10

clean:
	rm -rf artifacts/ build/ dist/ *.egg-info .pytest_cache
