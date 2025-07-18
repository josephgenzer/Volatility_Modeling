# Stock Price Volatility Forecasting
This project forecasts daily volatility of the S&P 500 (SPY), implementing and comparing classical statistical techniques like GARCH against machine learning methods such as LSTM and SVR. A hybrid model is then developed using a convex combination of the three forecasts, optimized for QLIKE loss, achieving better predictive performance than any individual model over the 5-year period from 2019 to 2024.

## EDA
- Squared log return plot exhibits clear volatility clustering, indicating conditional heteroskedasticity; this supports models with time-varying conditional variance, such as GARCH, and motivates nonlinear sequence models like LSTM or SVR to capture persistent volatility dynamics.
- ADF test confirm the log returns are stationary (TS ~ -10.5, p ~ 0), satisying the key assumption of weak stationarity required for GARCH modeling, promoting stable learning behavior for supervised ML models. 
- Volatility plots: Use 21-day rolling volatility as the baseline proxy for realized volatility, providing a simple target for model training and evaluation.
- Raw returns show no significant autocorrelation, while squared returns exhibit strong, persistent autocorrelation, condirming conditional heteroskedasticity, supporting the use of GARCH-type models and sequence models that capture temporal dependence in volatility dynamics.
- Log returns show heavy tails and non-normal distributions (JB p << 0.05), violating Gaussian error assumptions, motivating volatility models with fat-tailed innovations (GARCH with t or GED errors).
- Based on the presence of heavy tails and volatility clustering, QLIKE is a more appropriate evaluation metric than MSE for volatility forecasting models in this context.


## Model Selection and Results
- GARCH: A GARCH(1,1) model with Student’s t innovations was refit every day using a 250-day rolling window; parameters 
(𝜔,𝛼,𝛽) were estimated via maximum likelihood on each window.

- LSTM: An LSTM model was trained to predict 1-step-ahead volatility from 20-day input sequences of scaled rolling volatility using MSE as the training loss; hyperparameters (units, dropout, learning rate) were selected via grid search to minimize QLIKE on the test set.

- SVR: An SVR model with an RBF kernel was trained to predict 1-step-ahead volatility from 20-day lagged input vectors of scaled rolling volatility; hyperparameters (C, ε, γ) were selected via grid search to minimize QLIKE on the test set.

- Hybrid Model: A convex combination of the GARCH, LSTM, and SVR forecasts was used to construct a hybrid volatility forecast; optimal weights were computed by minimizing QLIKE on the test set subject to non-negativity and sum-to-one constraints.

## Future Direction

