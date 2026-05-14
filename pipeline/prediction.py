from utils.model_utils import load_registered_model
import logging


class predict_sentiment:
    def __init__(self):
        self.pipeline = load_registered_model("sentiment_model")
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, text):
        raw_results = self.pipeline(text)

        if raw_results and isinstance(raw_results[0], list):
            raw_results = raw_results[0]

        cleaned_results = []
        for item in raw_results:
            label = item.get("label", "")
            if label.startswith("LABEL_"):
                index = int(label.split("_")[-1])
                label = self.id2label.get(index, label)

            cleaned_results.append({
                "label": label,
                "score": float(item.get("score", 0.0)),
            })

        return cleaned_results
