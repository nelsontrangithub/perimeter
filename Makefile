.PHONY: install test test-invariants lint typecheck check bench bench-gate run build-admin generate-client generate-client-check admin-test container clean

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

generate-client:
	uv run python -m perimeter.server.openapi > admin/openapi.json
	cd admin && npm run generate

generate-client-check: generate-client
	git diff --exit-code -- admin/openapi.json admin/src/api/schema.d.ts

admin-test:
	cd admin && npm run typecheck && npm test

container:
	docker build -t perimeter:local .

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache .hypothesis admin/node_modules admin/dist
