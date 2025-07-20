# Stock Price Volatility Forecasting
Explored classical statistical and supervised ML models (GARCH, LSTM, SVR) to forecast daily volatility of the S&P 500 (SPY). Combined their outputs via a QLIKE-optimized convex combination, yielding a hybrid model that outperformed all individual models over the 2024–2025 period.

## EDA
- Squared log returns show volatility clustering, suggesting conditional heteroskedasticity, motivating models that capture time-varying volatility; explicitly in GARCH, nonparametrically via lagged inputs in LSTM and SVR
- ADF test confirms that log returns are weakly stationary (p ≈ 0), satisying GARCH assumptions and promoting stable learning for supervised ML models
- 21-day rolling volatility approximates monthly realized volatility and serves as a smooth, interpretable target for model training and evaluation
- Raw returns exhibit no autocorrelation, while squared returns exhibit strong persistence, indicating conditional heteroskedasticity and justifying models like GARCH
- Log returns have heavy tails and exhibit non-normality (JB p ≈ 0.05), violating Gaussian error assumptions, motivating the use of fat-tailed innovations in GARCH (e.g. Student's t)
- Given the heavy tails of log returns and volatility clustering, QLIKE is a more suitable evaluation metric than MSE; it is strictly consistent for conditional variance estimation and is used for hyperparameter selection in LSTM and SVR


## Model Experimentation
- GARCH(1,1)
  - Fit with Student’s t innovations, was refit daily using a 250-day rolling window
  - Parameters were estimated via maximum likelihood on each window
  - In the observed period, GARCH tracks medium-range volatility regimes well, but underreacts in periods of abrupt volatility reversal
 

- Long Short-Term Memory (LSTM)
  - An LSTM model was trained to predict 1-step-ahead volatility using 20-day sequences of scaled rolling volatility
  - Hyperparameters were selected via grid search to minimize QLIKE on the test set
  - Architecture: 1 LSTM layer (32 units), 0.2 dropout, dense output layer
  - Trained using MSE loss and the Adam optimizer (learning rate = 0.005) for 20 epochs, batch size = 32
  - In the observed period, LSTM closely tracks volatility during both calm and turbulent periods, including sharp regime shifts in mid-2024

- Support Vector Regression (SVR)
  - An SVR model was trained to predict 1-step-ahead volatility from 20-day lagged input vectors of scaled rolling volatility
  - Hyperparameters were selected via grid search to minimize QLIKE on the test set
  - Configuration: RBF kernel with C=10, epsilon=0.001, gamma='auto'
  - Trained using the default epsilon-insensitive loss with scikit-learn’s SVR implementation
  - In the observed period, SVR tracks both smooth and volatile phases, including the mid-2024 spike, while producing smoother forecasts than LSTM

- Hybrid Model
  - A convex combination of the GARCH, LSTM, and SVR forecasts was used to construct a hybrid volatility forecast
  - Weights were optimized to minimize QLIKE under non-negativity and sum-to-one constraints
    - Optimized weights: GARCH = 0.0742, LSTM = 0.4558, SVR = 0.4701
  - GARCH’s regime-level structure combined with the nonlinear adaptability of LSTM and SVR
  - In the observed period, the hybrid model balances smoothness and responsiveness, adapting effectively across volatility regimes including the mid-2024 spike

## Results
- Models were evaluated using QLIKE, MSE, Pearson correlation, and directional accuracy to assess distributional fit, pointwise error, co-movement, and trend alignment. SVR and LSTM outperformed GARCH by capturing nonlinear and sequential structures, while the hybrid model combined their strengths for more regime-adaptive forecast.
