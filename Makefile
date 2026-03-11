.PHONY: format lint test all docs-serve docs-build docs-deploy clean \
       download-baseline download-test download-all \
       train-baseline train-test train-all rclone-setup

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

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

download-baseline:
	uv run python scripts/download_data.py --baseline

download-test:
	uv run python scripts/download_data.py --test

download-all:
	uv run python scripts/download_data.py --all

# ---------------------------------------------------------------------------
# Batched training
# ---------------------------------------------------------------------------

train-baseline:
	uv run python scripts/train_batched.py --baseline

train-test:
	uv run python scripts/train_batched.py --test

train-all:
	uv run python scripts/train_batched.py --all

# ---------------------------------------------------------------------------
# rclone one-time setup
# ---------------------------------------------------------------------------

rclone-setup:
	bash scripts/configure_rclone.sh
