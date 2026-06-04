# ShopEase Sentiment Analysis System

An end-to-end NLP sentiment analysis project for an ecommerce platform called **ShopEase**. The system classifies customer reviews as **negative**, **neutral**, or **positive**, tracks experiments with **MLflow on DagsHub**, serves predictions through a **FastAPI** backend, and provides an interactive **Streamlit** dashboard for single-review and batch predictions.

## Project Highlights

- Fine-tunes a Transformer model for multilingual customer review sentiment classification.
- Tracks metrics, parameters, artifacts, and registered models with MLflow.
- Uses DagsHub as the remote MLflow tracking server.
- Provides a production-style FastAPI service with single and batch prediction endpoints.
- Includes a Streamlit dashboard for non-technical users.
- Supports CSV batch scoring for customer review datasets.
- Separates the project into data, training, prediction, API, dashboard, and utility layers.

## Business Problem

Ecommerce teams receive large volumes of customer reviews across products, countries, and languages. Manually reading every review is slow and inconsistent. This project helps automate review sentiment classification so business teams can:

- Monitor customer satisfaction.
- Identify negative feedback quickly.
- Compare sentiment across products or regions.
- Support customer experience and product improvement decisions.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| NLP Model | Hugging Face Transformers, DistilBERT multilingual |
| ML Framework | PyTorch, Transformers Trainer |
| Experiment Tracking | MLflow, DagsHub |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Data Processing | Pandas, scikit-learn, NLTK, spaCy |
| Version Control | Git, GitHub |

## Repository Structure

```text
10Alytics_sentiment_analysis_project/
├── config/
│   └── constant.py              # Project paths, model name, training arguments
├── Data/
│   ├── raw_reviews.csv          # Source review dataset
│   ├── testing_data.csv         # Working sample dataset
│   └── cleaned_review.csv       # Cleaned data output
├── main/
│   └── app.py                   # FastAPI application
├── pipeline/
│   ├── prediction.py            # Registered model loading and prediction logic
│   └── training.py              # End-to-end training pipeline
├── src/
│   ├── data_ingestion.py        # Data loading
│   ├── data_cleaning.py         # Text cleaning and label creation
│   ├── data_preprocessing.py    # Tokenization and dataset preparation
│   ├── model_training.py        # Model training and evaluation
│   └── model_pusher.py          # MLflow logging and model registration
├── utils/
│   └── model_utils.py           # MLflow helper utilities
├── streamlit_app.py             # Streamlit dashboard
├── requirements.txt             # Project dependencies
├── .env.example                 # Example environment configuration
└── README.md
```

## ML Workflow

1. Load customer review data.
2. Clean review text and map ratings to sentiment labels.
3. Split data into training and test sets.
4. Tokenize reviews using the Hugging Face tokenizer.
5. Fine-tune a sequence classification model.
6. Evaluate using accuracy, loss, and F1 score.
7. Log metrics and parameters to MLflow.
8. Register the best model in MLflow.
9. Serve the registered model through FastAPI.
10. Use Streamlit for interactive predictions.

## Sentiment Label Mapping

| Rating | Sentiment | Label |
| --- | --- | --- |
| 1-2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4-5 | Positive | 2 |

## MLflow and DagsHub

The project logs training runs to DagsHub-hosted MLflow.

- Experiment name: `10Alytics_sentiment_analysis`
- Registered model name: `sentiment_model`
- Tracked metrics: accuracy, loss, F1 score, previous best F1
- Tracked parameters: epochs, batch size, base model, improvement status

DagsHub MLflow URL:

```text
https://dagshub.com/onyedikakenechukwu7/10Alytics_sentiment_analysis_project.mlflow/#/experiments
```

## Setup Instructions

Clone the repository:

```powershell
git clone https://github.com/onyedika39/10Alytics_sentiment_analysis_project.git
cd 10Alytics_sentiment_analysis_project
```

Create and activate a virtual environment:

```powershell
python -m venv sentiment_env
.\sentiment_env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the FastAPI App

Start the API:

```powershell
uvicorn main.app:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
  "message": "Shopeease sentiment API is running"
}
```

## Prediction Example

Endpoint:

```text
POST /predict
```

Request body:

```json
{
  "text": "This product is excellent and I love it"
}
```

Example response:

```json
{
  "label": "positive",
  "confidence": 0.4783
}
```

## Batch Prediction

Endpoint:

```text
POST /predict/file
```

Upload a CSV file with a column named:

```text
review
```

The API returns the original rows with predicted sentiment labels and confidence scores.

## Running the Streamlit Dashboard

Keep the FastAPI server running in one terminal, then open a second terminal:

```powershell
.\sentiment_env\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

The dashboard supports:

- Single review prediction
- CSV batch prediction
- Triggering model retraining

## Training and Updating MLflow

To retrain the model from the terminal:

```powershell
python -c "from pipeline.training import Train_model; Train_model()"
```

Or use the FastAPI endpoint:

```text
GET /train
```

After training completes, refresh the DagsHub MLflow experiment page to see the updated run.

## Current Results

One tracked run produced:

| Metric | Value |
| --- | --- |
| Accuracy | 0.70 |
| F1 Score | 0.5765 |
| Loss | 0.8042 |

These results are based on a small dataset and should be interpreted as a proof of concept rather than a final production benchmark.

## Key Learnings

- Built an end-to-end NLP workflow from data ingestion to deployment.
- Integrated MLflow experiment tracking and model registration.
- Served a registered ML model through an API.
- Built a user-facing dashboard for real-time predictions.
- Practiced production-style project organization and documentation.

## Future Improvements

- Increase dataset size and balance negative, neutral, and positive classes.
- Track macro F1 and class-level precision/recall.
- Use raw text or lighter preprocessing for Transformer training.
- Add Docker support for reproducible deployment.
- Add automated tests for API endpoints and preprocessing.
- Add CI/CD workflow for linting and test execution.
- Deploy the API and Streamlit dashboard to a cloud platform.

## Author

**Michael Kenechukwu**  
NLP and Machine Learning Project Portfolio
