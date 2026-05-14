from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.data_ingestion import data_ingestion
import re
import logging
from config.constant import Clean_data

logging.basicConfig(level=logging.INFO)



class DataCleaning:

    def __init__(self):
        self.ensure_nltk()
    
    def _load_nlp(self) -> spacy.language.Language:
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                nlp = spacy.load(model)
                return nlp
            except OSError:
                continue

        nlp_fallback = spacy.blank("xx")
        return nlp_fallback
    

    def ensure_nltk(self) -> None:
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")

        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            try:
                nltk.download("punkt_tab")
            except Exception:
                pass

    
    def clean_text(self, text: str) -> str:
        """
        Convert text to lowercase, remove special characters,
        remove extra spaces, and return cleaned text.
        """

        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def lemmatize(self, text: str) -> str:
        """
        Convert words to their base form.
        Example: running -> run
        """

        nlp = self._load_nlp()
        doc = nlp(text)

        return " ".join([token.lemma_ for token in doc])

    def remove_stop_words(self, text: str) -> str:
        """
        Remove common words like the, is, in, and.
        """

        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]

        return " ".join(tokens)


def clean_data(data: pd.DataFrame):
    try:
        cleaner = DataCleaning()

        data["clean_text"] = data["review"].apply(cleaner.clean_text)

        data["lemma_text"] = data["clean_text"].apply(cleaner.lemmatize)

        data["final_text"] = data["lemma_text"].apply(cleaner.remove_stop_words)

        data["label"] = data["rating"].apply(
            lambda r: 0 if r in (1, 2) else (1 if r == 3 else 2)
        )
        data = data[["review", "final_text", "label"]]
        data.to_csv(Clean_data)

        logging.info("Dataset has been cleaned successfully")

        print(data.head())

        return data

    except Exception as e:
        logging.error(f"error occurred while cleaning data {e}")


