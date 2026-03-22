# Movie Recommender System

In this project, I built a movie recommendation engine using the MovieLens 1M dataset — a collection of one million movie ratings from over 6,000 users across nearly 4,000 films. The challenge is that the data is extremely "sparse" (users have rated less than 5% of available movies), so the system needs to infer preferences from limited information. I trained two fundamentally different models: SVD (a classical linear algebra approach) and Neural Collaborative Filtering (a deep learning approach), tuned both with Bayesian optimization, and evaluated them on both rating prediction accuracy and recommendation quality. The key finding is that these two goals are very different — SVD predicts star ratings more accurately (RMSE of 0.98), but NCF is dramatically better at actually recommending movies people will enjoy, with a 62% hit rate compared to SVD's 2%.

---

## Dataset Overview

The MovieLens 1M dataset is a benchmark dataset for recommender system research. It contains ratings from real users on a 1-5 star scale, along with demographic information about users and metadata about movies.

| Property | Value |
|----------|-------|
| Total Ratings | 1,000,209 |
| Unique Users | 6,040 |
| Unique Movies | 3,706 |
| Rating Scale | 1-5 stars (integers) |
| Mean Rating | 3.58 |
| Median Rating | 4.0 |
| Date Range | April 2000 - February 2003 |
| Sparsity | 95.53% |

The 95.53% sparsity means that for every user-movie pair in the matrix, 95 out of 100 cells are empty — the user never rated that movie. This is the fundamental challenge of recommendation: predicting how someone will feel about something they have never tried.

---

## Exploratory Data Analysis

### Rating Distribution

![Rating Distribution](01_eda/output/rating_distribution.png)

This bar chart shows how users distribute their ratings. The distribution skews positive — 4 stars is the most common rating, and the mean is 3.58. Very few users give 1-star ratings. This positive skew is typical of rating systems (people tend to watch movies they expect to enjoy) and affects model training because there are fewer examples of disliked movies to learn from.

### Ratings Over Time

![Ratings Over Time](01_eda/output/ratings_over_time.png)

This dual-axis chart shows the volume of ratings (bars) and the average rating (line) by month across the dataset period. Rating volume fluctuates significantly — with large spikes likely corresponding to when the platform actively solicited ratings — while the average rating remains remarkably stable around 3.5-3.6, suggesting that user sentiment is consistent regardless of when they rate.

### User Activity Distribution

![User Activity Distribution](01_eda/output/user_activity_distribution.png)

This histogram (on a logarithmic scale) shows how many ratings each user has contributed. The distribution follows a power law: most users have rated a moderate number of movies, but a few "super-raters" have rated hundreds or even thousands. Users with very few ratings are harder to make recommendations for because there is less data to learn their preferences — this is known as the "cold start" problem.

### Movie Popularity Distribution

![Movie Popularity Distribution](01_eda/output/movie_popularity_distribution.png)

Similarly, this histogram shows how many ratings each movie has received. A small number of popular blockbusters receive thousands of ratings, while the "long tail" of niche films has only a handful. Recommender systems need to balance recommending popular movies (which are safe bets) against surfacing lesser-known films that match a user's specific taste (which provides more value).

### Genre Analysis

![Genre Analysis](01_eda/output/genre_analysis.png)

This two-panel chart shows genre popularity (left) and average rating by genre (right). Drama and Comedy are the most common genres by volume. Film-Noir and Documentary receive the highest average ratings despite being much rarer — this suggests that users who seek out niche genres tend to rate them more generously, or that these genres simply produce higher-quality films on average.

### Demographic Analysis

![Demographic Analysis](01_eda/output/demographic_analysis.png)

These three panels break down rating behavior by user demographics. The gender panel shows the rating distribution for male and female users. The age panels reveal that older users tend to rate slightly higher on average and that the 25-34 age group contributes the most ratings by volume. Understanding these demographic patterns helps contextualize the model's predictions.

### User-Movie Interaction Heatmap

![Interaction Heatmap](01_eda/output/interaction_heatmap.png)

This heatmap shows ratings from the 20 most active users against the 20 most popular movies, with actual ratings annotated in each cell. Even among the most active users and popular movies, many cells are empty — illustrating the sparsity challenge. The filled cells show a range of ratings, confirming that even popular movies receive diverse opinions.

---

## Data Splitting Strategy

![Split Summary](02_split_data/output/split_summary.png)

I split the data chronologically — training on earlier ratings and testing on later ones — to simulate how a recommender system would work in production (it can only learn from past behavior to predict future preferences).

| Split | Ratings | Percentage | Date Range | Mean Rating | Users | Movies |
|-------|---------|-----------|------------|-------------|-------|--------|
| Training | 500,104 | 50% | Apr 2000 - Oct 2000 | 3.60 | 3,255 | 3,551 |
| Validation | 250,052 | 25% | Oct 2000 - Nov 2000 | 3.59 | 2,104 | 3,515 |
| Test | 250,053 | 25% | Nov 2000 - Feb 2003 | 3.53 | 2,035 | 3,541 |

The left panel confirms that average ratings are consistent across all three splits (~3.5-3.6), and the right panel shows the 50/25/25 split ratio. The chronological split is critical — random splitting would allow the model to "peek" at future ratings during training, producing overly optimistic results that would not hold up in production.

---

## Model 1: SVD (Singular Value Decomposition)

SVD is a classical linear algebra technique that decomposes the large, sparse user-movie rating matrix into smaller matrices that capture latent (hidden) factors. Think of it as automatically discovering the underlying dimensions that explain why people like certain movies — perhaps one dimension captures "how much someone likes action," another captures "preference for indie films," and so on. The model then predicts a rating by combining a global average, user-specific bias (some people rate everything high), movie-specific bias (some movies are universally loved), and the interaction between the user's and movie's latent factor profiles.

I used Bayesian optimization (Optuna, 30 trials) to tune the number of latent factors, finding that **k = 10** factors provided the best validation performance.

### SVD Prediction Distribution

![SVD Predictions](03_svd/output/prediction_distribution.png)

This chart compares the distribution of SVD's predicted ratings against the actual ratings. The predicted distribution tends to cluster more tightly around the mean (3.5-4.0) compared to the actual ratings, which spread more widely across the 1-5 scale. This "regression to the mean" is characteristic of matrix factorization methods — the model is cautious and avoids extreme predictions.

---

## Model 2: Neural Collaborative Filtering (NCF)

NCF uses a neural network to learn the relationship between users and movies. Instead of the linear decomposition used by SVD, NCF learns non-linear patterns through two parallel pathways: a Generalized Matrix Factorization (GMF) pathway that mirrors SVD's approach but in a neural framework, and a Multi-Layer Perceptron (MLP) pathway that can capture complex, non-linear interaction patterns. The outputs of both pathways are combined for the final prediction.

I tuned five hyperparameters using Bayesian optimization (Optuna, 5 trials):

| Parameter | Best Value | What It Controls |
|-----------|-----------|-----------------|
| Embedding Dimension | 93 | Size of each user/movie representation vector |
| Hidden Layers | 3 (93 → 46 → 23) | Depth and width of the MLP pathway |
| Dropout Rate | 3.7% | Fraction of neurons randomly disabled during training to prevent overfitting |
| Learning Rate | 0.00456 | Step size for gradient descent optimization |
| Batch Size | 1,024 | Number of ratings processed per training step |

### NCF Training History

![NCF Training Loss](04_neural_collaborative_filtering/output/training_loss_curve.png)

This chart tracks the model's error during training. The training loss (blue) steadily decreases as the model learns, while the validation loss (orange) plateaus and begins to rise slightly — a signal that the model is starting to memorize the training data rather than learning generalizable patterns. Early stopping automatically halted training at the optimal point before overfitting could degrade performance.

### NCF Prediction Distribution

![NCF Predictions](04_neural_collaborative_filtering/output/prediction_distribution.png)

Compared to SVD, NCF's predicted rating distribution more closely matches the shape of the actual ratings — it is more willing to predict extreme values (very high or very low ratings). This wider prediction range is one reason NCF excels at ranking: it creates more separation between movies a user would love versus merely like.

---

## Model Comparison

### Metrics Comparison

![Metrics Comparison](05_comparison/output/metrics_comparison.png)

This two-panel chart reveals the central finding of the project. The left panel shows rating prediction metrics (lower is better) — SVD wins with a lower RMSE and MAE. The right panel shows ranking/recommendation metrics (higher is better) — NCF dominates by enormous margins. This split performance demonstrates that rating prediction and recommendation quality are fundamentally different tasks.

**Rating Prediction Metrics (Test Set):**

| Metric | SVD | NCF | Winner |
|--------|-----|-----|--------|
| RMSE | 0.976 | 1.028 | SVD |
| MAE | 0.773 | 0.843 | SVD |

**Ranking/Recommendation Metrics (Test Set, K=10):**

| Metric | SVD | NCF | Improvement |
|--------|-----|-----|-------------|
| Precision@10 | 0.26% | 16.9% | 64x |
| Recall@10 | 0.08% | 3.4% | 45x |
| NDCG@10 | 0.016 | 0.371 | 23x |
| MAP@10 | 0.002 | 0.094 | 44x |
| Hit Rate@10 | 1.9% | 62.0% | 31x |
| Coverage | 1.7% | 29.6% | 17x |

- **Hit Rate@10**: The most intuitive metric — "if I show a user their top 10 recommendations, how often will at least one be a movie they actually rated highly?" NCF achieves 62% compared to SVD's 2%.
- **NDCG@10**: Measures whether the best recommendations appear at the top of the list. NCF scores 0.371 versus SVD's 0.016.
- **Coverage**: The percentage of the movie catalog that gets recommended to at least one user. NCF recommends 29.6% of movies versus SVD's 1.7%, meaning NCF surfaces a much more diverse set of recommendations.

### Prediction Distribution Overlay

![Prediction Overlay](05_comparison/output/prediction_overlay.png)

This overlay shows both models' prediction distributions against the actual ratings on the same plot. SVD (blue) produces a narrow, concentrated distribution centered around 3.5, while NCF (orange) spreads wider to better match the actual distribution (red dashed). The wider spread gives NCF more "dynamic range" to differentiate between movies a user would love versus those they would only moderately enjoy.

### Error Distribution

![Error Distribution](05_comparison/output/error_distribution.png)

These panels compare the prediction errors of both models. The left panel (density plot) shows that SVD's errors are more tightly centered around zero, while NCF's errors spread slightly wider. The right panel (box plot) confirms this — SVD's error box is more compact. Despite these larger rating errors, NCF's superior ranking performance shows that getting the exact star rating right matters less than getting the relative ordering right.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Chronological data split | Prevents data leakage by ensuring models only learn from past ratings, matching how a real recommender would operate. |
| Both rating and ranking metrics | RMSE alone would declare SVD the winner, hiding the fact that NCF is 31x better at actual recommendations. Evaluating both reveals the full picture. |
| Bayesian hyperparameter tuning | Optuna's intelligent search finds good parameters in 5-30 trials, far more efficient than exhaustive grid search over the large neural network parameter space. |
| Two complementary architectures | SVD represents the classical linear approach, while NCF represents the modern deep learning approach. Comparing them reveals when complexity is justified — for ranking, it clearly is. |
| User and item bias terms in SVD | Some users rate generously and some movies are universally popular. Explicitly modeling these biases improves rating prediction and creates a strong baseline for comparison. |
| Early stopping for NCF | Neural networks can memorize training data if trained too long. Early stopping monitors validation performance and halts training at the optimal point, preventing overfitting. |
