# DLMDSPWP01 Dataset Project

Python implementation for the dataset-based final project of the module `DLMDSPWP01 – Programming with Python`.

## Purpose

- Assignment-aligned behavior and schemas
- Deterministic outputs
- Reproducible runs from validated CSV inputs
- Clear module boundaries and tests

## Dataset inputs

Place the following CSV files in `dataset/`:

- `train.csv`
- `ideal.csv`
- `test.csv`

## Technology stack

- Python
- `pandas`
- `SQLAlchemy`
- `SQLite`
- `Bokeh`
- `pytest`

## Repository layout

- `dataset/`: source CSV files
- `src/`: application code (`dlmdspwp01_project`)
- `tests/`: unit and integration tests
- `artifacts/db/`: generated SQLite database (ignored except `.gitkeep`)
- `artifacts/plots/`: generated Bokeh HTML plots (ignored except `.gitkeep`)
- `artifacts/reports/`: CSV/JSON summaries (ignored except `.gitkeep`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running

Default paths:

```bash
dlmdspwp01-project
```

Or:

```bash
python3 -m dlmdspwp01_project.main
```

Override paths:

```bash
dlmdspwp01-project \
  --train dataset/train.csv \
  --ideal dataset/ideal.csv \
  --test dataset/test.csv \
  --database artifacts/db/assignment.db \
  --plots-dir artifacts/plots \
  --reports-dir artifacts/reports
```

## Testing

```bash
pytest
```

## Generated artifacts

After a run, typical outputs are:

- `artifacts/db/assignment.db`
- `artifacts/plots/training_vs_selected_ideal.html`
- `artifacts/plots/mapped_test_points.html`
- `artifacts/plots/mapping_distribution.html`
- `artifacts/reports/selection_summary.csv`
- `artifacts/reports/mapped_test_points.csv`
- `artifacts/reports/mapping_counts_by_function.csv`
- `artifacts/reports/mapping_summary.csv`
- `artifacts/reports/run_summary.json`

## Report files

- `selection_summary.csv`: selected ideal per training function, SSE, max deviation, threshold.
- `mapped_test_points.csv`: successfully mapped test points only.
- `mapping_counts_by_function.csv`: mapped count per selected ideal.
- `mapping_summary.csv`: totals (mapped / unmapped).
- `run_summary.json`: paths and counts for automation.
