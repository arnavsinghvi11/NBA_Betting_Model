# NBA Sports Betting Predictor 🏀

## Project Overview
I developed a machine learning sports betting predictor for the 2022-23 NBA season focused on optimizing player bet predictions based on past game performances. This project was primarily to assess the model's theoretical performance rather than profit generation.

The existing sports betting landscape often favors betting sportsbooks due to their undisclosed parameters, causing a disparity against casual bettors. By harnessing data, I aimed to optimize bet predictions to level this playing field.

## Data Collection & Preprocessing
I collected expert predictions from sports betting websites that produce daily outputs through their proprietary analyzers. These expert predictions served as inputs for my optimization model. With past game statistics from the ongoing season obtained from online game box scores, I conducted extensive data processing and cleaning to develop a **daily statistics evaluator** for determining whether a player achieved the bet prop's threshold and a **real-time prediction scraper** to produce my model's insights before game times.

## Model Optimization
To optimize the predictions:

1. **Ensemble Learning**: Utilized various ML models (logistic regression, LDA, KNN, Gaussian NB, SVM, MLP, decision trees, forests, AdaBoost, gradient boosting, deep neural networks) to create an ensemble based on expert predictions. This considered features like past expert success rate, specified stat type, betting odds, and more.
2. **Multiplicative Weights Algorithm**: Here, every ML model (MLP, DNN, RF, XGB, LSTM, and the aforementioned Online Experts Ensemble Learning) acts as its own expert. They predict player stats using advanced features, and these predictions are weighted based on their past accuracy. This method rewards models that are accurate and penalizes those that aren't.

## Results
While the average "educated" bettor typically achieves about a 50% success rate on NBA bets, my model showcased a peak accuracy of ~62% for the 2022-23 NBA season. Several strategies consistently achieved over 54% accuracy. It's vital to understand that these results come with the underlying consideration of variability with the volume of bets.

Below is a breakdown of various methodologies and their outcomes:

| Methodology | Approach Description | Win-Loss Record | Success Rate | Profit per Bet (ROI) |
|-------------|----------------------|-----------------|--------------|----------------|
| 1. Optimized Online Predictions | All predictions | 754-591 | 56.06% | 3.21% |
| | Regular season success rate > 0.5 | 458-308 | 59.79% | 7.65% |
| | Regular season success rate > 0.5 and minimum 1 unit | 221-145 | **60.38%** | **11.08%** |
| 2. Multiplicative Weighted Experts for Last 5 Games Performance (Intersection) | All predictions | 258-176 | 59.45% | 4.96% |
| | Regular season success rate > 0.5 | 145-89 | **61.97%** | **8.29%** |
| | Regular season success rate > 0.5 and minimum 1 unit | 68-43 | 61.26% | 7.96% |
| 3. Multiplicative Weighted Experts for Last 10 Games Performance (Intersection) | All predictions | 246-205 | 54.55% | -3.19% |
| | Regular season success rate > 0.5 | 169-122 | **58.08%** | **2.72%** |
| | Regular season success rate > 0.5 and minimum 1 unit | 73-55 | 57.03% | 1.98% |
| 4. Multiplicative Weighted Experts for Last 5 Games Performance (Combination) | All predictions | 361-305 | 54.20% | -1.07% |
| | Regular season success rate > 0.5 | 204-140 | **59.30%** | **5.98%** |
| | Regular season success rate > 0.5 and minimum 1 unit | 93-66 | 58.49% | 4.26% |
| 5. Multiplicative Weighted Experts for Last 10 Games Performance (Combination) | All predictions | 363-289 | 55.67% | -0.38% |
| | Regular season success rate > 0.5 | 246-168 | **59.42%** | **4.88%** |
| | Regular season success rate > 0.5 and minimum 1 unit | 101-75 | 57.39% | 3.51% |


**Note:** Bold values represent the highest performers in their respective methodology brackets.

## Legend

### Methodologies:

1. **Optimized Online Predictions**:
   - Optimizes the scraped online bet predictions via an ensemble voting classifier of the top 5 models.
   - Considers factors like prediction type, expert behind the prediction, betting odds, payout, profit, and more.
   - Find daily predictions in the `NBA-Bets-Optimized-Predictions` folder.

2. **Multiplicative Weighted Experts**:
   - An approach where each ML model serves as an "expert" predicting player stats.
   - Prediction weights adjust based on historical accuracy.
   - Variants and their locations include:
     - **Last 5 Games Performance**: Draws training data from the recent 5 games of a player. Access predictions at `NBA-Bets-Multiplicative-Weights/weights_past_5_games/{month}/{day}/`.
     - **Last 10 Games Performance**: Draws training data from the recent 10 games of a player. Access predictions at `NBA-Bets-Multiplicative-Weights/weights_past_10_games/{month}/{day}/`.
     - **Intersection**: Considers predictions agreed upon by all experts. Files are labeled `intersection_{prop_type}.csv` in their respective folders.
     - **Combination**: Aggregates predictions from all experts. Files are labeled `combination_{prop_type}.csv` in their respective folders.

### Filters:
- **Regular season success rate > 0.5**: Filters predictions to those where a player achieved the target stat threshold in over half of all games leading up to the upcoming game.
- **Minimum 1 unit**: Filters predictions to those with bets placed for 1 unit or higher, excluding volatile lower unit predictions with higher odds and payouts but lower success rates.
