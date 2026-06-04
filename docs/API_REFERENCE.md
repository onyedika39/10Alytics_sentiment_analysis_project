# API Reference

This project exposes a FastAPI backend for sentiment prediction and model retraining.

## Base URL

```text
http://127.0.0.1:8000
```

## Health Check

```text
GET /
```

Response:

```json
{
  "message": "Shopeease sentiment API is running"
}
```

## Single Review Prediction

```text
POST /predict
```

Request body:

```json
{
  "text": "This product is excellent and I love it"
}
```

Response:

```json
{
  "label": "positive",
  "confidence": 0.4783
}
```

## Batch Review Prediction

```text
POST /predict/file
```

Upload a CSV file containing a `review` column.

Response:

```json
{
  "results": [
    {
      "review": "The product arrived late.",
      "sentiment_label": "negative",
      "sentiment_confidence": 0.72
    }
  ]
}
```

## Trigger Training

```text
GET /train
```

This endpoint starts model training, evaluates the model, and logs the run to MLflow/DagsHub.

Response:

```json
{
  "message": "Model trained successfully"
}
```

## Notes

- `/predict` loads the registered MLflow model lazily on the first request.
- `/train` can take time because it runs the training pipeline.
- Predictions do not create new MLflow runs; only training updates MLflow.
