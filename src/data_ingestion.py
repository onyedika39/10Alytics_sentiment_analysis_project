import pandas as pd
import logging
from config.constant import input_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def data_ingestion():
    try:
        data = pd.read_csv(input_data)
        logging.info("Data loaded successfully loaded...")
        return data
    except Exception as e:
        logging.exception(f"error occured while loading dataset: {e}")
        raise


if __name__ == "__main__":
    print(data_ingestion().head())
