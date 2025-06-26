# Implicit Feedback Recommendation System

## Overview

This is a Streamlit-based web application that implements an implicit feedback recommendation system using matrix factorization algorithms. The system is designed to work with implicit user interaction data (views, clicks, purchases) rather than explicit ratings. It provides a complete pipeline from data processing to model training, evaluation, and recommendation generation.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **UI Structure**: Multi-page navigation with sidebar
- **Pages**: Data Overview, Model Training, Generate Recommendations, Evaluation, Performance Metrics
- **Visualization**: Plotly for interactive charts and graphs
- **Session Management**: Streamlit session state for maintaining application state

### Backend Architecture
- **Core Components**: Modular Python classes for different functionalities
- **Data Processing**: `DataProcessor` class handles user-item matrix creation and preprocessing
- **Recommendation Engine**: `ImplicitRecommender` class wraps multiple matrix factorization algorithms
- **Evaluation Framework**: `RecommendationEvaluator` class provides comprehensive model evaluation
- **Utilities**: Helper functions for data generation and formatting

### Algorithm Support
- **Alternating Least Squares (ALS)**: Default algorithm for collaborative filtering
- **Bayesian Personalized Ranking (BPR)**: For learning from implicit feedback
- **Logistic Matrix Factorization (LMF)**: Alternative matrix factorization approach

## Key Components

### 1. Data Processor (`data_processor.py`)
- Creates sparse user-item interaction matrices
- Handles user and item ID mapping
- Implements data filtering based on minimum interaction thresholds
- Manages train/test data splitting

### 2. Recommendation Engine (`recommender.py`)
- Supports multiple implicit feedback algorithms (ALS, BPR, LMF)
- Configurable hyperparameters (factors, regularization, iterations)
- Provides recommendation generation for individual users
- Handles model training and persistence

### 3. Evaluation Framework (`evaluator.py`)
- Implements standard recommendation metrics (Precision@K, Recall@K, NDCG)
- Supports evaluation across multiple K values
- Provides comprehensive model performance assessment
- Handles cold-start user scenarios

### 4. Utilities (`utils.py`)
- Synthetic data generation for demonstration purposes
- Data formatting and visualization helpers
- Logging configuration and management

### 5. Main Application (`app.py`)
- Streamlit web interface coordination
- Session state management
- Page routing and navigation
- Integration of all components

## Data Flow

1. **Data Input**: Users can upload interaction data or generate synthetic data
2. **Preprocessing**: Data is filtered, mapped, and converted to sparse matrices
3. **Model Training**: Selected algorithm trains on the user-item interaction matrix
4. **Evaluation**: Model performance is assessed using standard metrics
5. **Recommendation Generation**: Trained model generates personalized recommendations
6. **Visualization**: Results are displayed through interactive Plotly charts

## External Dependencies

### Core Libraries
- **implicit**: Matrix factorization algorithms for implicit feedback
- **scipy**: Sparse matrix operations and scientific computing
- **scikit-learn**: Machine learning utilities and evaluation metrics
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing operations

### Visualization and UI
- **streamlit**: Web application framework
- **plotly**: Interactive data visualization

### Development Tools
- **uv**: Python package management
- **logging**: Application logging and debugging

## Deployment Strategy

### Platform
- **Target**: Replit autoscale deployment
- **Runtime**: Python 3.11 with Nix package management
- **Port**: Application runs on port 5000

### Configuration
- **Deployment Target**: Autoscale for dynamic resource allocation
- **Package Management**: UV for dependency resolution and installation
- **Environment**: Nix-based reproducible environment

### Workflow
- **Development**: Parallel workflow execution
- **Dependency Installation**: Automatic package installation on startup
- **Server Launch**: Streamlit server with specified port configuration

## Changelog

- June 26, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.