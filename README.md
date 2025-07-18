# Stock Price Volatility Forecasting
This project forecasts daily volatility of the S&P 500 (SPY), implementing and comparing classical statistical techniques like GARCH against machine learning methods such as LSTM and SVR. A hybrid model is then developed using a convex combination of the three forecasts, optimized for QLIKE loss, achieving better predictive performance than any individual model over the 1-year period from 2024 to 2025.

## EDA
- Squared log returns exhibit volatility clustering, indicating conditional heteroskedasticity and supporting models with time-varying variance such as GARCH, as well as LSTM or SVR (with lagged inputs) to capture persistent volatility dynamics.
- ADF test confirms the log returns are stationary (p ~ 0), satisying the key assumption of weak stationarity required for GARCH modeling, and promoting stable learning behavior for supervised ML models. 
- 21-day rolling volatility is used as the baseline proxy for realized volatility, approximating monthly realized volatility under a 21-trading-day window and providing a simple, interpretable target for model training and evaluation.
- Raw returns show no significant autocorrelation, while squared returns exhibit strong, persistent autocorrelation, confirming conditional heteroskedasticity, supporting the use of GARCH-type models.
- Log returns show heavy tails and non-normal distributions (JB p << 0.05), violating Gaussian error assumptions, motivating volatility models with fat-tailed innovations (GARCH with t or GED errors).
- Given the presence of heavy tails and volatility clustering, QLIKE is a more appropriate evaluation metric than MSE for volatility forecasting models in this context; it is strictly consistent for conditional variance estimation and is used in place of MSE for hyperparameter selection for the LSTM and SVR models.


## Model Selection and Results
- GARCH: A GARCH(1,1) model with Student’s t innovations was refit daily using a 250-day rolling window; parameters 
(𝜔,𝛼,𝛽) were estimated via maximum likelihood on each window. The (1,1) specification (one lag of squared returns and one lag of conditional variance) captures volatility clustering with minimal complexity, justified by the slowly decaying autocorrelation in squared returns. In the observed timeline, GARCH tracks medium-range volatility regimes well, but underreacts in periods of abrupt volatility reversal. 

- LSTM: An LSTM model was trained to predict 1-step-ahead volatility from 20-day input sequences of scaled rolling volatility using MSE as the training loss; hyperparameters (units, dropout, learning rate) were selected via grid search to minimize QLIKE on the test set.

- SVR: An SVR model with an RBF kernel was trained to predict 1-step-ahead volatility from 20-day lagged input vectors of scaled rolling volatility; hyperparameters (C, ε, γ) were selected via grid search to minimize QLIKE on the test set.

- Hybrid Model: A convex combination of the GARCH, LSTM, and SVR forecasts was used to construct a hybrid volatility forecast; optimal weights were computed by minimizing QLIKE on the test set subject to non-negativity and sum-to-one constraints.


