import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import logging
import os
import torch
from src.data_cleaning import clean_data
from config.constant import model_name, train_data, test_data
from src.data_ingestion import data_ingestion

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class data_processor:
    def __init__(self):
        self.data = data_ingestion()
        self.clean_data = clean_data(self.data)

    def split_data(self):
        try:
            X = self.clean_data["final_text"].astype(str)
            y = self.clean_data["label"]
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
            logging.info("Dataset has been split into training and testing sets successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.exception(f"error occurred while splitting data {e}")
            raise


class Tokenizerwrapper:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):
        return self.tokenizer(text.tolist(), truncation=True, padding=True, max_length=128)


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        if hasattr(labels, "tolist"):
            self.labels = labels.tolist()
        elif hasattr(labels, "__iter__") and not isinstance(labels, (list, tuple)):
            self.labels = list(labels)
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def prepare_sentiment_data():
    try:
        processor = data_processor()
        X_train, X_test, y_train, y_test = processor.split_data()
        tokenizer = Tokenizerwrapper(model_name)
        train_encodings = tokenizer.encode(X_train)
        test_encodings = tokenizer.encode(X_test)

        train_dataset = SentimentDataset(train_encodings, y_train)
        test_dataset = SentimentDataset(test_encodings, y_test)

        os.makedirs(os.path.dirname(train_data), exist_ok=True)
        os.makedirs(os.path.dirname(test_data), exist_ok=True)
        torch.save(train_dataset, train_data)
        torch.save(test_dataset, test_data)
        logging.info("data have been prepared and ready to be passed into the model..")
        return train_dataset, test_dataset
    except Exception as e:
        logging.exception(f"error occurred while preparing and processing the dataset {e}")
        raise


if __name__ == "__main__":
    prepare_sentiment_data()
