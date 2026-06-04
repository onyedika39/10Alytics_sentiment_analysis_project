# Model Card

## Model Overview

This project fine-tunes a multilingual Transformer model for ecommerce customer review sentiment classification.

## Base Model

```text
distilbert-base-multilingual-cased
```

## Task

Text classification with three sentiment classes:

- Negative
- Neutral
- Positive

## Label Mapping

| Rating | Sentiment | Label |
| --- | --- | --- |
| 1-2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4-5 | Positive | 2 |

## Intended Use

The model is intended for classifying ecommerce review sentiment and supporting customer experience analysis.

Example use cases:

- Identify negative customer reviews.
- Monitor product satisfaction.
- Score batches of customer feedback.
- Support product and operations teams with review insights.

## Not Intended For

The model should not be used as the only basis for high-stakes business decisions. It is a portfolio-grade proof of concept and should be validated on larger, more representative datasets before production use.

## Evaluation Metrics

Tracked in MLflow:

- Accuracy
- F1 score
- Evaluation loss

One logged run produced:

| Metric | Value |
| --- | --- |
| Accuracy | 0.70 |
| F1 Score | 0.5765 |
| Loss | 0.8042 |

## Limitations

- Dataset is small and class distribution is imbalanced.
- Confidence scores may be low for some obvious examples.
- Current preprocessing may remove useful sentiment signals.
- More evaluation is needed across countries, product categories, and languages.

## Recommended Improvements

- Add more balanced data for negative and neutral reviews.
- Track macro F1 and class-level precision/recall.
- Compare raw text against cleaned text for Transformer training.
- Add confusion matrix analysis.
- Add automated tests and model validation checks.
