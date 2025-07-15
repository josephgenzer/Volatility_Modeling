# Stock Price Volatility Forecasting
This project forecasts daily volatility of the S&P 500 (SPY), implementing and comparing classical statistical techniques like GARCH against machine learning methods such as LSTM and SVR. A hybrid model is then developed using a convex combination of the three forecasts, optimized for QLIKE loss, achieving better predictive performance than any individual model over the 5-year period from 2019 to 2024.

## EDA
- Squared log return plot exhibits clear volatility clustering, suggesting models that capture heteroskedasticity and nonlinear temporal dependencies
- ADF test: Strong rejection of the null hypothesis (unit root) indicates that log returns are stationary, which is a required assumption for GARCH and improves learning stability for supervised learning models with time-series data.
- Volatility plots: Use 21-day rolling volatility as the baseline proxy for realized volatility, providing a simple target for model training and evaluation.
- ACF Plots: Unlike the autocorrelation of raw returns (which decay immediately), the squared returns exhibit significant autocorrelation across a range of lags, indicating that past volatility has predictive power.
- Distributional Analysis: Log returns are non-normally distributed, and instead have heavy tails with a negative skew. This justifies using student-t errors in GARCH and nonparametric models (LSTM, SVR). _____ also indicates the use of QLIKE loss instead of MSE for model evaluation and training.


## Model Selection and Results
- GARCH(1,1)

- LSTM

- SVR

- Hybrid Model

## Future Direction

