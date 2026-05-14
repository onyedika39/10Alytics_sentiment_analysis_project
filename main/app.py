import io
import logging
from fastapi import FastAPI, UploadFile, File
import pandas as pd
from pipeline.training import Train_model
from pipeline.prediction import predict_sentiment
from pydantic import BaseModel

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="shopease sentiment system api")
_predictor = None


class TextRequest(BaseModel):
    text: str


def get_predictor():
    global _predictor
    if _predictor is None:
        logging.info("Loading registered sentiment model from MLflow...")
        _predictor = predict_sentiment()
        logging.info("Model loaded successfully")
    return _predictor


@app.get("/")
def home():
    return {"message": "Shopeease sentiment API is running"}


@app.post("/predict")
def predict_user_sentiment(request: TextRequest):
    try:
        predictor = get_predictor()
        results = predictor.predict(request.text)
        top_label = max(results, key=lambda x: x["score"])
        return {
            "label": top_label["label"],
            "confidence": float(top_label["score"]),
        }
    except Exception as e:
        logging.exception(f"error occurred during prediction: {e}")
        return {"error": str(e)}


@app.post("/predict/file")
async def predict_batch_sentiment(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    if "review" not in df.columns:
        return {"error": "CSV file must contain a 'review' column."}

    predictor = get_predictor()
    result_list = []
    for idx, row in df.iterrows():
        try:
            review = str(row["review"])
            results = predictor.predict(review)
            if results is None or len(results) == 0:
                raise ValueError("Empty result from the model")
            top_label = max(results, key=lambda x: x["score"])
            result_row = row.to_dict()
            result_row["sentiment_label"] = top_label["label"]
            result_row["sentiment_confidence"] = float(top_label["score"])
            result_list.append(result_row)
        except Exception as e:
            logging.exception(f"error for review index {idx}: {e}")
            result_row = row.to_dict()
            result_row["sentiment_label"] = "error"
            result_row["sentiment_confidence"] = 0.0
            result_list.append(result_row)

    return {"results": result_list}


@app.get("/train")
def train_model():
    try:
        Train_model()
        return {"message": "Model trained successfully"}
    except Exception as e:
        logging.exception(f"error occurred during model training: {e}")
        return {"error": str(e)}
