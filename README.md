# AIML-Recruitment-2026--ADITYA_K-


- **Name:** Aditya K
- **Registration No:** RA2511003011165
- **Mail:** ak1427@srmist.edu.in
- **Degree:** B.Tech CSE

## 2. Tasks Completed

- **Task 1:** Air Quality Forecasting (`CO(GT)` and `C6H6(GT)`)

## 3. Problem Statement

- **Task 1:** Forecast `CO(GT)` and `C6H6(GT)` concentration levels six hours into the future using hourly sensor data from an Italian city. A primary objective was establishing a rigirous validation pipeline to prevent data leakage to reflect true forecasting viability.


## 4. Approach

- **Data Cleaning:** Replaced `-200` missing-value sentinels with `NaN` and strictly limited forward-filling to gaps of 3 hours or less to prevent fabricating data.
- **Feature Engineering:** Applied cyclical encoding (sine/cosine) to `Hour` and `DayOfWeek`. Excluded `Month` and `Season` to prevent the model from memorizing the single year of available data.
- **Validation:** Enforced strict chronological splitting (80/20 by timestamp) rather than row-shuffling to ensure the target future column (`shift(-6)`) never leaked into the training set.
- **Modeling:** Trained and evaluated Linear Regression, Ridge, Random Forest, and XGBoost using a walk-forward `TimeSeriesSplit`. Models were trained on both raw targets and `log1p(target)`, then converted back to original scale (`expm1`) for a fair error comparison.


## 5. Technologies Used

- **Languages:** Python
- **Libraries (Task 1):** Pandas, NumPy, Scikit-learn (RandomForestRegressor, Ridge, TimeSeriesSplit, RandomizedSearchCV), XGBoost, Matplotlib, Seaborn.


## 6. Results



- **CO(GT):** The Random Forest (log1p target) model achieved an $R^2$ of **0.46** on the untouched test set, outperforming the strongest Hour-of-week baseline ($R^2$ = **0.40**).
- **C6H6(GT):** Forecasted with an $R^2$ of **0.38**, demonstrating that its 0.98 contemporaneous correlation with the proxy sensor does not equate to strong 6-hour forecasting power.
- **Ablation:** Removing current weather and secondary pollutant features improved generalizability, indicating the leanest model (cyclical time + target history) handled the noise best.

## 7. Key Learnings

1. **High Correlation $\neq$ Forecasting Power:** `C6H6(GT)` and its proxy sensor correlate at 0.98 in the same hour, but forecasting it 6 hours ahead scored lower ($R^2$ 0.38) than `CO(GT)` ($R^2$ 0.46), proving immediate sensor relationships do not guarantee predictive horizons.

2. **Chronological Splitting for Time Series:** An experimental leaky setup using a shuffled split inflated the $R^2$ artificially to 0.93. The 0.47-point gap between the leaky split and the honest chronological forecast highlights exactly how severely random splits compromise time-series evaluation.

## 8. Challenges

- **Handling Massive Missing Data Spans:** Specifically, 1,683 missing hours for `CO(GT)`, including one 173-hour consecutive gap.
  - **Solution:** I avoided the temptation to use `bfill()` or long-range interpolation, which would inject future data backward into the timeline. Instead, I built custom grouping logic to measure gap lengths, forward-filled only those $\leq$ 3 hours, and dropped the remaining missing rows safely *after* establishing the strict calendar split date.
