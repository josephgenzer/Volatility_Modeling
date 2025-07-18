# Stock Price Volatility Forecasting
Explored classical statistical and supervised ML models (GARCH, LSTM, SVR) to forecast daily volatility of the S&P 500 (SPY), ultimately developing a convex combination-based hybrid model optimized for QLIKE loss, which outperformed all individual models over the 2024-2025 period.

## EDA
- Squared log returns show volatility clustering, suggesting conditional heteroskedasticity and supporting models with time-varying variance such as GARCH, as well as LSTM and SVR (with lagged inputs) to capture persistence in volatility.
- ADF test confirms the stationarity of log returns (p ≈ 0), satisying the weak stationarity assumption required for GARCH and promoting stable learning behavior for supervised ML models. 
- 21-day rolling volatility is used as a proxy for realized volatility, approximating a monthly window and providing a simple, interpretable target for model training and evaluation.
- Raw returns show no autocorrelation, while squared returns exhibit strong persistence, reinforcing the use of models that capture conditional heteroskedasticity (eg GARCH).
- Log returns show heavy tails and non-normality (JB p ≈ 0.05), violating Gaussian error assumptions, motivating fat-tailed innovations in GARCH (eg Student's t).
- QLIKE is a more suitable evaluation metric than MSE due to the heavy tails and volatility clustering; it is strictly consistent for conditional variance estimation and is used for hyperparameter selection in LSTM and SVR.


## Model Experimentation
- GARCH
  - A GARCH(1,1) model with Student’s t innovations was refit daily using a 250-day rolling window. Parameters were estimated via maximum likelihood on each window. The (1,1) specification (one lag of squared returns and one lag of conditional variance) captures volatility clustering with minimal complexity, justified by persistent autocorrelation in squared returns.
  - In our observed timeline, GARCH tracks medium-range volatility regimes well, but underreacts in periods of abrupt volatility reversal. 
- Long Short-Term Memory (LSTM)
  - An LSTM model was trained to predict 1-step-ahead volatility using 20-day input sequences of scaled rolling volatility. Model was with MSE loss for stability, while hyperparameters (units, dropout, learning rate) were selected via grid search to minimize QLIKE on the test set.
  - LSTM captures nonlinear temporal dependencies in volatility, which are not accounted for by parametric models like GARCH
  - In our observed timeline, LSTM adapts more flexibly to changes in volatility but exhibits lag in sharp regime transitions, likely due to its smoothing behavior

- Support Vector Regression (SVR)
  - An SVR model with an RBF kernel was trained to predict 1-step-ahead volatility from 20-day lagged input vectors of scaled rolling volatility. Hyperparameters were selected via grid search to minimize QLIKE on the test set.
  - SVR captures nonlines patterns in volatility without sequential memory, yielding a flexible, nonparametric alternative to GARCH.
  - In out observed timeline, SVR appears to smooth short-term fluctuations more aggressively than LSTM

- Hybrid Model
  - A convex combination of the GARCH, LSTM, and SVR forecasts was used to construct a hybrid volatility forecast. Optimal weights were computed by minimizing QLIKE on the test set subject to non-negativity and sum-to-one constraints.

## Results
