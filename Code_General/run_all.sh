#!/usr/bin/env bash
# Regenerates everything from scratch (extension 44). Usage: bash run_all.sh [python]   (default: python)
# Order: data -> libraries -> tests -> notebooks 01-16 (cached blocks resume). Notebook 07-08 trainings take hours; see CLAUDE.md.
set -euo pipefail
PY=${1:-python}; cd "$(dirname "$0")"
$PY download_data.py && $PY download_coinc_data.py && $PY download_events.py
$PY imr_tools.py                                  # needs lalsuite
$PY -m pytest tests -q                           # needs pytest
for nb in 0[1-9]_*.ipynb 1[0-6]_*.ipynb; do
  echo "== $nb"; $PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=${KERNEL:-python3} "$nb"
done
$PY make_summary_figures.py
