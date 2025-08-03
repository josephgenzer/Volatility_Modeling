# Stock Price Volatility Forecasting
Explored classical statistical, supervised ML, and deep learning models (GARCH, LSTM, SVR) to forecast daily volatility of the S&P 500 (SPY). Combined their outputs via a QLIKE-optimized convex combination, yielding a hybrid model that outperformed all individual models over the 2024–2025 period.

## EDA
- Squared log returns show volatility clustering, suggesting conditional heteroskedasticity, motivating models that capture time-varying volatility; explicitly in GARCH, nonparametrically via lagged inputs in LSTM and SVR
- ADF test confirms that log returns are weakly stationary (p ≈ 0), satisying GARCH assumptions and promoting stable learning for supervised ML models
- 21-day rolling volatility approximates monthly realized volatility and serves as a smooth, interpretable target for model training and evaluation
- Raw returns exhibit no autocorrelation, while squared returns exhibit strong persistence, indicating conditional heteroskedasticity and justifying models like GARCH
- Log returns have heavy tails and exhibit non-normality (JB p ≈ 0), violating Gaussian error assumptions, motivating the use of fat-tailed innovations in GARCH (e.g. Student's t)
- Given the heavy tails of log returns and volatility clustering, QLIKE is a more suitable evaluation metric than MSE; it is strictly consistent for conditional variance estimation and is used for hyperparameter selection in LSTM and SVR


## Model Experimentation
- GARCH(1,1)
  - Fit with Student’s t innovations, was refit daily using a 250-day rolling window
  - Parameters were estimated via maximum likelihood on each window
  - Captures medium-range volatility regimes but underreacts during sharp regime shifts
 
- Long Short-Term Memory (LSTM)
  - Trained to predict 1-step-ahead volatility using 20-day sequences of scaled rolling volatility
  - Hyperparameters selected via grid search to minimize QLIKE on the test set
  - Architecture: 1 LSTM layer (32 units), 0.2 dropout, dense output layer
  - Trained using MSE loss and the Adam optimizer (learning rate = 0.005) for 20 epochs, batch size = 32
  - Tracks volatility well across calm and turbulent periods, including sharp reversals in mid-2024

- Support Vector Regression (SVR)
  - Trained to predict 1-step-ahead volatility from 20-day lagged input vectors of scaled rolling volatility
  - Hyperparameters were selected via grid search to minimize QLIKE on the test set
  - Configuration: RBF kernel with C=10, epsilon=0.001, gamma='auto'
  - Trained using the default epsilon-insensitive loss with scikit-learn’s SVR implementation
  - Produces smoother forecasts than LSTM while retaining responsiveness during volatility spikes

- Hybrid Model
  - A convex combination of the GARCH, LSTM, and SVR forecasts was used to construct a hybrid volatility forecast
  - Weights were optimized to minimize QLIKE under non-negativity and sum-to-one constraints
    - Optimized weights: GARCH = 0.07, LSTM = 0.46, SVR = 0.47
  - Combines GARCH’s regime structure with LSTM/SVR's nonlinear adaptability
  - Balances smoothness and responsiveness, adapting well to dynamic regimes including the 2024 spike

- Models evaluated on QLIKE, MSE, Pearson correlation, and directional accuracy to capture distributional fit, pointwise error, co-movement, and trend alignment. LSTM and SVR outperformed GARCH by capturing sequential and nonlinear patterns. The hybrid model achieved the best overall performance by combining the complementary strengths of each approach, offering a more robust and adaptive volatility forecast.
