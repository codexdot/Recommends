# Recommends: Implicit Feedback Recommendation System

---

## Project Links

- [GitHub Repository](https://github.com/codexdot/Recommends)
- Live on: 
---

## Overview

**Recommends** is a modular, production-ready Streamlit web application for building, training, and evaluating recommendation systems based on implicit user feedback (such as views, clicks, and purchases). It supports multiple matrix factorization algorithms and provides a full workflow from data ingestion to model evaluation and recommendation visualization.

---

## Features

- **End-to-End Pipeline**: Data loading, preprocessing, model training, evaluation, and recommendation generation.
- **Algorithm Support**: Alternating Least Squares (ALS), Bayesian Personalized Ranking (BPR), Logistic Matrix Factorization (LMF).
- **Web UI**: Streamlit-based, multi-page navigation, interactive visualizations (Plotly).
- **Synthetic & Custom Data**: Generate demo data or upload your own (CSV).
- **Comprehensive Evaluation**: Standard metrics (Precision@K, Recall@K, NDCG), cold-start handling.
- **Model Management**: Save, load, and package full model artifacts.
- **Deployment Ready**: Designed for Replit autoscale, Python 3.11.

---

## System Architecture

### Frontend

- **Framework**: Streamlit
- **Pages**: Data Overview, Model Training, Generate Recommendations, Evaluation, Performance Metrics
- **Visualization**: Plotly for interactive charts
- **Session State**: Streamlit session management for continuity

### Backend

- **Data Processing**: `DataProcessor` class (matrix creation, ID mapping, splitting)
- **Recommendation Engine**: `ImplicitRecommender` (algorithm wrapper, training, recommendations)
- **Evaluation**: `RecommendationEvaluator` (metrics, multi-K, cold-start)
- **Cold Start Handling**: `ColdStartHandler` (user/item clustering, popular/diverse/trending detection)
- **Model Manager**: `ModelManager` (save/load artifacts)
- **Utilities**: Data generation, formatting, logging

---

## Data Flow

1. **Data Input**: Upload CSV or generate synthetic (see below for format)
2. **Preprocessing**: Filter, map, convert to sparse matrices
3. **Model Training**: Train ALS/BPR/LMF with configurable hyperparameters
4. **Evaluation**: Assess performance with multiple metrics
5. **Recommendation**: Generate personalized recommendations
6. **Visualization**: Results shown via interactive charts

---

## Usage

### 1. Running the App

```
streamlit run app.py
```
- Or, deploy on [Replit](https://replit.com/) (see `.replit` for autoscale config).

### 2. Data Format

CSV file must include columns:  
- `user_id`: Unique identifier for user  
- `item_id`: Unique identifier for item  
- `rating`: Implicit feedback score (e.g., 1 for view, 2 for click, etc.)

### 3. Navigating the UI

- **Data Overview**: Upload or generate data, visualize basic stats
- **Model Training**: Select algorithm, configure hyperparameters, train model
- **Generate Recommendations**: View recommendations for users
- **Evaluation**: See metrics (Precision@K, Recall@K, NDCG@K)
- **Performance Metrics**: Visualize and compare model performance

---

## Key Modules

- `app.py`: Main Streamlit UI and routing
- `data_processor.py`: Data ingestion, filtering, matrix creation
- `recommender.py`: Algorithms (ALS, BPR, LMF)
- `evaluator.py`: Metrics and evaluation logic
- `cold_start_handler.py`: Cold start user/item management
- `model_manager.py`: Save/load model packages (including mappings and evaluation results)
- `utils.py`: Demo data generation, helpers

---

## Algorithms Supported

- **ALS (Alternating Least Squares)**: Standard collaborative filtering
- **BPR (Bayesian Personalized Ranking)**: Pairwise ranking for implicit signals
- **LMF (Logistic Matrix Factorization)**: Probabilistic approach

---

## Dependencies

- Python 3.11
- `implicit`
- `scipy`
- `scikit-learn`
- `pandas`
- `numpy`
- `streamlit`
- `plotly`
- (See `uv.lock` for full list)

---

## Deployment

- **Replit**: Configured for autoscale deployment (`.replit`, `.streamlit/config.toml`)
- **Local**: Install dependencies, run with Streamlit

---

## Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/foo`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## License

This project is open source and available under the MIT License.


---
