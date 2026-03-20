.PHONY: help setup install test test-unit test-integration test-memory test-rules test-agent lint format type-check clean run run-research

help:
	@echo "Local Research Agent - Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup              Create virtual environment and install deps"
	@echo "  make install            Install dependencies"
	@echo "  make install-dev        Install dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test               Run all tests"
	@echo "  make test-unit          Run unit tests only"
	@echo "  make test-integration   Run integration tests"
	@echo "  make test-memory        Run memory layer tests"
	@echo "  make test-rules         Run rules engine tests"
	@echo "  make test-agent         Run agent orchestrator tests"
	@echo "  make test-coverage      Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint               Run linters (ruff)"
	@echo "  make format             Format code (black)"
	@echo "  make type-check         Run type checker (mypy)"
	@echo "  make check              Run all checks (lint, type, tests)"
	@echo ""
	@echo "Development:"
	@echo "  make clean              Remove build artifacts and cache"
	@echo "  make run                Run agent CLI"
	@echo "  make run-research       Run example research task"
	@echo ""

setup:
	python3 -m venv venv
	. venv/bin/activate && make install-dev
	@echo "✅ Virtual environment ready: source venv/bin/activate"

install:
	pip install -q -r requirements.txt

install-dev:
	pip install -q -r requirements.txt
	pip install -q black ruff mypy pytest-cov

test:
	pytest tests/ -v --asyncio-mode=auto

test-unit:
	pytest tests/ -v -m "not integration" --asyncio-mode=auto

test-integration:
	pytest tests/test_integration.py -v --asyncio-mode=auto

test-memory:
	pytest tests/test_memory.py -v -m memory --asyncio-mode=auto

test-rules:
	pytest tests/test_rules.py -v -m rules --asyncio-mode=auto

test-agent:
	pytest tests/test_agent.py -v -m agent --asyncio-mode=auto

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term --asyncio-mode=auto
	@echo "✅ Coverage report: open htmlcov/index.html"

lint:
	ruff check src tests scripts

format:
	black src tests scripts --line-length 100

type-check:
	mypy src --ignore-missing-imports

check: lint type-check test

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name "htmlcov" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build dist *.egg-info

run:
	python -m scripts.agent --help

run-research:
	python -m scripts.agent run \
		--objective "Research the latest advances in local LLM inference" \
		--max-steps 10 \
		--rules config/rules.yaml

.DEFAULT_GOAL := help
