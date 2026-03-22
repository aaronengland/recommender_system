# Recommender System

A production-grade movie recommender system built on the MovieLens 1M dataset, demonstrating both classical and cutting-edge collaborative filtering techniques.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [0. Data Collection](#0-data-collection)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Data Splitting](#2-data-splitting)
  - [3. SVD Matrix Factorization](#3-svd-matrix-factorization)
  - [4. Neural Collaborative Filtering](#4-neural-collaborative-filtering)
  - [5. Model Comparison](#5-model-comparison)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Design Decisions](#key-design-decisions)
- [Getting Started](#getting-started)

---

## Overview

This project builds and compares two recommender system approaches on the MovieLens 1M dataset:

- **End-to-end pipeline** from data collection through model comparison
- **Chronological (out-of-time) train/validation/test split** to simulate real-world deployment
- **Classical baseline**: Truncated SVD with bias terms, implemented from scratch using `scipy.sparse.linalg.svds`
- **Cutting-edge model**: Neural Collaborative Filtering (NCF) combining GMF and MLP pathways in PyTorch
- **Bayesian hyperparameter optimization** with Optuna for both models
- **Best-practice evaluation** with both rating prediction metrics (RMSE, MAE) and ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K, Coverage)
- **All data stored in S3** as Parquet files; only models, plots, and small summary tables stored locally

---

## Project Structure

```
recommender_system/
├── 00_data_collection/
│   ├── notebook.ipynb          # Download MovieLens 1M, parse, upload to S3
│   └── output/
├── 01_eda/
│   ├── notebook.ipynb          # Exploratory data analysis
│   └── output/                 # 7 visualization PNGs
├── 02_split_data/
│   ├── notebook.ipynb          # Chronological train/valid/test split
│   └── output/                 # Split summary PNG
├── 03_svd/
│   ├── notebook.ipynb          # SVD matrix factorization
│   └── output/                 # Model artifacts + tuning CSV + predictions
├── 04_neural_collaborative_filtering/
│   ├── notebook.ipynb          # Neural Collaborative Filtering (PyTorch)
│   └── output/                 # Model artifacts + tuning CSV + predictions
├── 05_comparison/
│   ├── notebook.ipynb          # Head-to-head model comparison
│   └── output/                 # Comparison visualizations
├── requirements.txt
└── README.md
```

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) |
| Ratings | ~1,000,000 |
| Users | ~6,000 |
| Movies | ~4,000 |
| Rating Scale | 1 - 5 (integer) |
| Time Range | April 2000 - February 2003 |
| Features | User demographics (gender, age, occupation), movie genres |
| Storage | `s3://recommender-system-demo/` (Parquet format) |

---

## Methodology

### 0. Data Collection

Downloads the MovieLens 1M dataset from GroupLens, parses the `::` delimited `.dat` files (ratings, users, movies), converts timestamps to datetime, and uploads three Parquet files to S3. No data is stored locally.

### 1. Exploratory Data Analysis

Comprehensive exploration of the dataset including:

- **Rating distribution** across the 1-5 scale
- **Temporal patterns** in rating volume and mean rating over time
- **User activity distribution** (ratings per user, log-scale)
- **Movie popularity distribution** (ratings per movie, log-scale)
- **Genre analysis** (frequency and mean rating by genre)
- **Demographic analysis** (rating patterns by gender, age group)
- **User-movie interaction heatmap** visualizing the sparse interaction matrix

### 2. Data Splitting

Chronological (out-of-time) split maintaining temporal order:

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 50% | Model fitting |
| Validation | 25% | Hyperparameter tuning & early stopping |
| Test | 25% | Final evaluation |

Ratings are sorted by timestamp and split sequentially, simulating the real scenario where a model is trained on historical interactions and evaluated on future ones.

### 3. SVD Matrix Factorization

Classical collaborative filtering baseline using truncated SVD decomposition:

- **Bias modeling**: Global mean + user bias + item bias
- **Matrix decomposition**: `scipy.sparse.linalg.svds` on the mean-centered user-item matrix
- **Prediction**: `global_mean + user_bias + item_bias + U * Sigma * Vt`
- **Tuning**: Optuna optimizes the number of latent factors (k=10-200) over 30 trials
- **Cold start handling**: Falls back to bias terms for unknown users/items

### 4. Neural Collaborative Filtering

Deep learning recommender combining two parallel pathways:

- **GMF (Generalized Matrix Factorization)**: Element-wise product of user/item embeddings
- **MLP (Multi-Layer Perceptron)**: Concatenated user/item embeddings through hidden layers with ReLU and dropout
- **NeuMF**: GMF and MLP outputs concatenated and passed through a final prediction layer
- **Training**: Adam optimizer with MSE loss and early stopping (patience=5)
- **Tuning**: Optuna optimizes embedding dimension, MLP architecture, dropout, learning rate, and batch size over 20 trials

### 5. Model Comparison

Head-to-head comparison including:

- **Metrics table** with all rating and ranking metrics for both models
- **Grouped bar charts** for rating prediction metrics and ranking metrics
- **Prediction distribution overlay** comparing predicted rating distributions
- **Error distribution comparison** with KDE and box plots

---

## Evaluation Metrics

### Rating Prediction (computed on all test interactions)

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |

### Ranking (computed per-user, then averaged; K=10)

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K | Fraction of relevant items found in top-K |
| NDCG@K | Normalized Discounted Cumulative Gain (position-aware) |
| MAP@K | Mean Average Precision |
| Hit Rate@K | Fraction of users with at least 1 relevant item in top-K |
| Coverage | Fraction of catalog items recommended across all users |

**Relevance threshold**: rating >= 4.0 (standard in recommender systems literature)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| MovieLens 1M dataset | Large enough to be meaningful (~1M ratings), small enough for fast iteration. Well-established benchmark in recommender systems research. |
| Chronological split (not random) | Random splits leak future interactions into training, inflating metrics unrealistically. Chronological splits simulate real deployment. |
| `scipy.sparse.linalg.svds` (not Surprise library) | Demonstrates understanding of the underlying linear algebra rather than relying on a high-level API. |
| PyTorch for NCF (not TensorFlow) | Industry standard for ML research and increasingly for production. Shows modern deep learning competency. |
| Combined GMF + MLP architecture | The NeuMF architecture from He et al. (2017) combines the strengths of linear (GMF) and non-linear (MLP) interaction modeling. |
| Both rating and ranking metrics | Rating prediction (RMSE/MAE) alone is insufficient. Recommender systems are fundamentally ranking problems, so Precision@K, NDCG@K, etc. are essential. |
| Relevance threshold of 4.0 | Standard in literature. Ratings of 4-5 indicate genuine interest; 3 is ambiguous; 1-2 are negative signals. |
| K=10 for @K metrics | Industry standard reflecting realistic top-of-page recommendation slots. |
| Parquet format for data storage | More efficient than CSV for columnar data: better compression, type preservation, and faster I/O. |
| Self-contained notebooks | Each notebook defines its own classes and can run independently on SageMaker without shared module imports. |
| Optuna for hyperparameter tuning | Bayesian optimization is more sample-efficient than grid search or random search, finding better hyperparameters in fewer trials. |
| Early stopping for NCF | Prevents overfitting without requiring manual epoch selection. Patience of 5 epochs balances convergence and training time. |

---

## Getting Started

### Prerequisites

- Python 3.8+
- AWS credentials configured with access to S3 bucket `recommender-system-demo`
- (Optional) CUDA-compatible GPU for faster NCF training

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

Execute notebooks in order on AWS SageMaker:

1. `00_data_collection/notebook.ipynb` - Downloads data and uploads to S3
2. `01_eda/notebook.ipynb` - Explores the dataset
3. `02_split_data/notebook.ipynb` - Creates train/valid/test splits
4. `03_svd/notebook.ipynb` - Trains SVD baseline
5. `04_neural_collaborative_filtering/notebook.ipynb` - Trains NCF model
6. `05_comparison/notebook.ipynb` - Compares both models
