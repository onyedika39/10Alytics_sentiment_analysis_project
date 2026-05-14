from src.data_preprocessing import prepare_sentiment_data
from src.model_training import ModelTraining
from src.model_pusher import ModelPusher
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def Train_model():
    try:
        train_dataset, test_dataset = prepare_sentiment_data()
        model = ModelTraining()
        trainer = model.model_training(train_dataset=train_dataset, test_dataset=test_dataset)
        results = model.model_evaluation(trainer)
        print(results)
        model_pusher = ModelPusher()
        model_pusher.updated_model_pusher(trainer, results)
        return trainer, results
    except Exception as e:
        logging.exception(f"error occurred during model training {e}")
        raise
