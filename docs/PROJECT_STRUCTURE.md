# Project Structure

This repository is organized to separate data processing, model training, model serving, and user-facing dashboard logic.

## Folders

| Folder | Purpose |
| --- | --- |
| `config/` | Central project constants, paths, model name, and training arguments |
| `Data/` | Input and prepared CSV data |
| `main/` | FastAPI application |
| `pipeline/` | End-to-end training and prediction workflows |
| `src/` | Core data ingestion, cleaning, preprocessing, training, and MLflow logging modules |
| `utils/` | Utility functions for MLflow model lookup and loading |
| `docs/` | Project documentation for API, model card, and structure |

## Main Files

| File | Purpose |
| --- | --- |
| `main/app.py` | Defines API endpoints for prediction, batch prediction, and training |
| `streamlit_app.py` | Interactive dashboard for users |
| `src/model_training.py` | Handles model training and evaluation |
| `src/model_pusher.py` | Logs metrics and model artifacts to MLflow |
| `pipeline/prediction.py` | Loads the registered MLflow model and returns predictions |
| `pipeline/training.py` | Runs the full training workflow |
| `utils/model_utils.py` | Gets best MLflow run and loads registered model |

## Ignored Generated Files

The following are intentionally excluded from GitHub:

- Virtual environments: `sentiment_env/`, `shopease_env/`
- Model checkpoints: `results/`
- Local MLflow runs: `mlruns/`
- Logs: `logs/`
- Tokenized torch datasets: `Data/processed_data/`

Large trained models and artifacts should be stored in MLflow/DagsHub instead of GitHub.
