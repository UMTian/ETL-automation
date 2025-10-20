# Universal ETL + ML Pipeline Dashboard

Local-first Streamlit app to generate data, run ETL, train ML models, and visualize preprocessing and performance.

## Quick start

```bash
# Windows PowerShell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt

streamlit run dashboard.py
```

## Operating the app

- Sidebar modes:
  - Data Upload: Load from `data/generated/` or upload CSV/JSON/Parquet. Data is sanitized for display.
  - ETL Pipeline: Pick data type (ecommerce/financial/healthcare/ml_training), sample size, output format; Run. Saves to `data/output/`.
  - ML Pipeline: Choose training data, task (classification/regression), models; Train. Results appear under Model Results.
  - Complete ETL + ML Pipeline: Generates data and trains models in one go.

- Main sections:
  - Pipeline Monitoring: Status, records processed, best model score.
  - Quick Statistics: Shape, memory, missing values, source counts, preview.
  - Data Visualization tabs:
    - Data Distribution: Histogram + stats for numeric columns.
    - Data Quality: Missing values chart and types table.
    - Preprocessing Steps: Configure a simulated pipeline and see step impacts.
    - Correlation Analysis: Heatmap, table, high-correlation pairs.
    - Sample Data: Adjustable head and dataset info.
  - Model Results: Metrics table and grouped bar chart; best model and score.

## Outcomes

- Visual, end-to-end control over ETL and ML workflows
- Immediate view of data quality and preprocessing effects
- Quick benchmarks of classic ML models on tabular data
- Saved datasets under `data/output/`

## Cleaning and preprocessing

- ETL cleaning: via `DataCleaner` (duplicates, missing values, outliers, dtype standardization) configured in YAML (`transform.cleaning_rules`).
- ML preprocessing: via `FeatureEngineer` (missing values, categorical encoding, scaling, optional feature selection) applied automatically before training.

## Supported tasks (now)

- Tabular classification/regression on synthetic datasets:
  - Ecommerce: high-value transaction classification; amount regression
  - Financial: fraud classification (simulated); amount regression
  - Healthcare: diagnosis presence classification; visit-duration regression
  - General ML training: binary classification target included

## Troubleshooting

- Arrow/pyarrow display errors: refresh, or re-upload; app sanitizes DataFrames prior to display.
- Launch warnings: use `streamlit run dashboard.py` (not `python dashboard.py`).
- Missing deps: `pip install -r requirements.txt` in the venv.

## Key files

- `dashboard.py`: Streamlit app
- `src/pipeline.py`: ETL pipeline
- `src/transformers/cleaner.py`: DataCleaner
- `src/ml/feature_engineering.py`: FeatureEngineer
- `src/ml/model_trainer.py`: Model training
- `src/models/`: Model factories and sklearn models
- `data/`: Generated output and uploads

## Project structure

```text
.
├─ dashboard.py
├─ requirements.txt
├─ README.md
├─ config.yaml                # Optional runtime config (see below)
├─ data/
│  ├─ generated/             # Built-in sample datasets
│  ├─ uploads/               # User-uploaded files
│  └─ output/                # ETL outputs and trained artifacts
└─ src/
   ├─ pipeline.py
   ├─ transformers/
   │  └─ cleaner.py
   ├─ ml/
   │  ├─ feature_engineering.py
   │  └─ model_trainer.py
   └─ models/
```

## Configuration (optional `config.yaml`)

Place a `config.yaml` in the project root to override defaults used by the app/ETL.

```yaml
data:
  input_dir: data/generated
  upload_dir: data/uploads
  output_dir: data/output

etl:
  dataset_type: ecommerce            # ecommerce | financial | healthcare | ml_training
  sample_size: 5000
  output_format: parquet             # csv | parquet | json
  cleaning_rules:
    drop_duplicates: true
    fill_missing:
      strategy: mean                 # mean | median | most_frequent | constant
      constant_value: null
    outlier:
      method: zscore
      z_threshold: 3.0

ml:
  task: classification               # classification | regression
  models:                            # subset of supported models, e.g. lr, rf, xgb
    - lr
    - rf
  training:
    cv_folds: 5
    random_state: 42
```

If no `config.yaml` is present, the UI controls determine these values at runtime.

## Common workflows

### 1) Use built-in sample data and run full pipeline

```bash
streamlit run dashboard.py
# Sidebar → Complete ETL + ML Pipeline → choose dataset + sizes → Run
```

### 2) Upload your own data and train

```bash
streamlit run dashboard.py
# Sidebar → Data Upload → pick CSV/JSON/Parquet → then ML Pipeline → Train
```

### 3) Generate ETL outputs only

```bash
streamlit run dashboard.py
# Sidebar → ETL Pipeline → configure → Run (outputs to data/output)
```

## Extending the system

- Add a new dataset type: create a generator under `src/` and wire it in `src/pipeline.py`.
- Customize cleaning: extend `DataCleaner` in `src/transformers/cleaner.py` and expose a rule in YAML/UI.
- Add a model: register a factory under `src/models/` and update `src/ml/model_trainer.py` choices.
- Modify preprocessing: adjust `FeatureEngineer` in `src/ml/feature_engineering.py`.

## Windows notes

- Prefer running in a virtual environment as shown in Quick start.
- If execution is blocked, allow the venv activation script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in an elevated PowerShell, then re-open the terminal.

## Support

If something breaks or you have a feature request, open an issue or leave a note in your project tracker. Include repro steps, dataset shape, and any console output.