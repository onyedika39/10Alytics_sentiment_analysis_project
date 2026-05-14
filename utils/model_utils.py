import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient
import dagshub


def get_best_model(experiment_name="10Alytics_sentiment_analysis"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        return None

    best_model = sorted(runs, key=lambda x: x.data.metrics.get("f1", 0), reverse=True)[0]
    return best_model


def get_best_f1(experiment_name="10Alytics_sentiment_analysis"):
    best_run = get_best_model(experiment_name)
    if best_run is not None:
        return best_run.data.metrics.get("f1", 0)
    return None


def load_registered_model(registered_model_name="sentiment_model"):
    dagshub.init(
        repo_owner="onyedikakenechukwu7",
        repo_name="10Alytics_sentiment_analysis_project",
        mlflow=True,
    )
    model_uri = f"models:/{registered_model_name}/latest"
    return mlflow.transformers.load_model(model_uri)


# Backward-compatible alias for older code.
def log_registered_model(registered_model_name="sentiment_model"):
    return load_registered_model(registered_model_name)
