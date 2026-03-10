.PHONY: format lint test all docs-serve docs-build docs-deploy clean

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest -p no:cacheprovider -m "not slow" -n auto

all: format lint test

docs-serve:
	uv run mkdocs serve --livereload

docs-build:
	uv run mkdocs build

docs-deploy:
	uv run mkdocs gh-deploy --force

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache dist/ site/
