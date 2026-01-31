# TORERO PITCHING

This repo contains Streamlit applications and modeling assets for pitching analytics.

## Project overview

So much of this repo is thanks to the great work of Thomas Nestico, Kyle Bland, and Max Bay for providing examples of models or other essential code in their GitHub repos.

This repo contains a Streamlit dashboard for USD baseball pitcher analysis. The app loads TrackMan pitch-by-pitch data, enriches it with derived metrics (pitch classification, release/movement metrics, count context, and Stuff+/whiff scores), and surfaces visuals for pitch design, command, and usage.

### What the Streamlit app shows

- **Pitch metrics table**: Per-pitcher averages for velocity, movement, release, and model outputs (Stuff+/whiff).
- **Pitch locations (RHH/LHH)**: Plate-location plots split by batter handedness.
- **Break plots**: Movement profiles and Stuff+ overlays to compare shapes by pitch type.
- **Release plots + command ellipses**: Release point clustering and confidence ellipses for command.
- **Stuff+ over time**: Rolling trends for model outputs by pitch type.
- **Tilt plots + ideal locations**: Polar plots for tilt and model-driven target locations.
- **Raw data view**: Filtered TrackMan rows for troubleshooting or export.

### Data inputs and expectations

- **Default data files**: The app reads TrackMan exports such as `data/raw/usd_baseball_TM_master_file.csv` and `data/raw/armangle_final_fall_usd.csv`.
- **Required columns**: The workflows rely on pitcher identifiers, pitch type labels, release metrics (speed, height, side), movement metrics, plate location, dates, and pitch counts (e.g., `Pitcher`, `Pitchtype`, `Relspeed`, `Relheight`, `Horzbreak`, `Inducedvertbreak`, `Platelocside`, `Platelocheight`, `Date`, `PitcherPitchNo`).
- **Report tables**: `pitcher_reports.py` uses percentile and run-value tables stored in `data/raw` (for example `pitch_metric_percentiles.csv`, `count_summary_table.csv`, and `run_values.csv`).

### Models and artifacts

- **Stuff+ and whiff models**: `models/NCAA_STUFF_PLUS_ALL.joblib` and `models/whiff_model_grouped_training.joblib` score pitch quality.
- **Run value model**: `models/rv_with_plateloc.joblib` supports run-value scoring in the full app.
- **Supporting assets**: static images used in reports live in `assets/`, with configuration defaults stored under `config/`.

## Project structure

```
.
├── assets/                 # Static assets (logos, images used in reports/apps)
│   └── images/
├── config/                 # Centralized configuration and environment templates
├── data/                   # Versioned datasets used by apps and reports
│   └── raw/                # Immutable raw inputs pulled into models/apps
├── docs/                   # Project documentation and design notes
├── models/                 # Trained model artifacts (joblib/json)
├── pitcher_reports.py      # Report generation module
├── streamlit_app.py        # Primary Streamlit app entrypoint
├── streamlit_app_full.py   # Extended Streamlit app entrypoint
└── requirements.txt
```

### Why each top-level directory exists

- **assets/**: Keeps visual assets separate from code/data, making packaging and deployment cleaner.
- **config/**: Houses configuration defaults and environment templates to avoid hard-coded paths.
- **data/**: Tracks raw inputs with version control and enforces a single source of truth.
- **docs/**: Central place for technical docs, model notes, and onboarding guides.
- **models/**: Stores model artifacts independently from code so they can be versioned or swapped.

## Naming conventions

- **Files**: use `snake_case` for CSVs and model artifacts (e.g., `armangle_final_fall_usd.csv`).
- **Folders**: use short, descriptive names (`data/raw`, `models`, `assets/images`).
- **Python modules**: use `snake_case` (`streamlit_app.py`, `pitcher_reports.py`).
- **Model artifacts**: prefix with model intent (`stuff_plus_*.joblib`) when adding new versions.

## Configuration management

- Prefer a `config/base.yaml` (checked in) and `config/local.yaml` (gitignored) for overrides.
- Use environment variables (e.g., `DATA_DIR`, `MODEL_DIR`) for production deployments.
- Store secrets in `.env` files that are excluded from git.

## How to run it on your own machine

1. Install the requirements

   ```
   pip install -r requirements.txt
   ```

2. Run the app

   ```
   streamlit run streamlit_app.py
   ```
