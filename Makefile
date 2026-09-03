.PHONY: install test lint typecheck check bench run build-admin container clean

install:
	uv sync --all-groups
	cd admin && npm ci

test:
	uv run pytest

test-invariants:
	uv run pytest -m invariant -v

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

check: lint typecheck test

bench:
	uv run python -m bench.run --out bench/results.md

bench-gate:
	uv run pytest -m bench -v -p no:cacheprovider

run:
	uv run perimeter serve

build-admin:
	cd admin && npm run build

container:
	docker build -t perimeter:local .

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache .hypothesis admin/node_modules admin/dist
