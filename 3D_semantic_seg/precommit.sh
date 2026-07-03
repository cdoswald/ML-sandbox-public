#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running Python compile check..."
python -m compileall src

echo "Running Ruff static linting..."
ruff check src

echo "Running Ruff formatting..."
ruff format src

echo "Running mypy type checking..."
mypy src

echo "All checks passed."