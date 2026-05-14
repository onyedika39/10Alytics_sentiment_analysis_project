import dagshub
import mlflow
import mlflow.transformers
import logging
from transformers import pipeline
from utils.model_utils import get_best_f1
from config.constant import training_args, model_name


class ModelPusher:
    def __init__(self):
        dagshub.init(
            repo_owner="onyedikakenechukwu7",
            repo_name="10Alytics_sentiment_analysis_project",
            mlflow=True,
        )
        self.experiment_name = "10Alytics_sentiment_analysis"
        mlflow.set_experiment(self.experiment_name)

    def updated_model_pusher(self, trainer, metrics):
        try:
            new_f1 = metrics.get("eval_f1", metrics.get("f1", 0))
            old_f1 = get_best_f1(self.experiment_name)
            model_improved = old_f1 is None or new_f1 > old_f1

            with mlflow.start_run(run_name="sentiment_model_training"):
                # Log every training attempt so it is visible in DagsHub MLflow.
                mlflow.log_metric("accuracy", metrics.get("eval_accuracy", metrics.get("accuracy", 0)))
                mlflow.log_metric("loss", metrics.get("eval_loss", metrics.get("loss", 0)))
                mlflow.log_metric("f1", new_f1)

                if old_f1 is not None:
                    mlflow.log_metric("previous_best_f1", old_f1)

                mlflow.log_param("epochs", training_args.num_train_epochs)
                mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
                mlflow.log_param("base_model", model_name)
                mlflow.log_param("model_improved", model_improved)

                if model_improved:
                    sentiment_pipeline = pipeline(
                        task="text-classification",
                        model=trainer.model,
                        tokenizer=model_name,
                        return_all_scores=True,
                    )

                    mlflow.transformers.log_model(
                        transformers_model=sentiment_pipeline,
                        artifact_path="model",
                        registered_model_name="sentiment_model",
                    )
                    logging.info("New model + tokenizer has been logged and registered successfully in MLflow")
                else:
                    logging.info("Training run logged to MLflow, but model was not registered because it did not improve.")
        except Exception as e:
            logging.exception(f"error occurred while pushing model to mlflow {e}")
