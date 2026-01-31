# TORERO PITCHING

This repo contains Streamlit applications and modeling assets for pitching analytics.

## Project structure (proposed + applied)

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

## README content recommendations (for future expansion)

- **Project overview**: short summary of the use case and data sources.
- **Quickstart**: setup, run, and environment requirements.
- **Data dictionary**: link to dataset schema and provenance.
- **Model registry**: list of models with training windows and metrics.
- **Operational runbook**: deployment steps and rollback guidance.

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
