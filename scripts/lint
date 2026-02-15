#!/usr/bin/env bash
# Runs ruff formatter

set -e

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "Running ruff format..."
ruff format .

echo "Running ruff check..."
ruff check . --fix

echo ""
echo "✅ All checks passed!"

