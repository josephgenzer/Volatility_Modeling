# Stock Price Volatility Forecasting
Explored classical statistical and supervised ML models (GARCH, LSTM, SVR) to forecast daily volatility of the S&P 500 (SPY). Combined their outputs using a convex combination optimized for QLIKE loss, resulting in a hybrid model that outperformed all individual models over the 2024–2025 period.

## EDA
- Squared log returns show volatility clustering, suggesting conditional heteroskedasticity, justifying models with time-varying variance such as GARCH, as well as LSTM and SVR (with lagged inputs) to capture persistence
- ADF test confirms that log returns are stationary (p ≈ 0), satisying the weak stationarity assumption for GARCH and supporting stable learning for supervised ML models
- 21-day rolling volatility serves as a proxy for realized volatility, approximating a monthly window and providing a simple, interpretable target for model training and evaluation
- Raw returns show no autocorrelation, while squared returns exhibit strong persistence, supporting the use of models that capture conditional heteroskedasticity, such as GARCH
- Log returns exhibit heavy tails and non-normality (JB p ≈ 0.05), violating Gaussian error assumptions, justifying fat-tailed innovations in GARCH (e.g. Student's t)
- Given the heavy tails of log returns and volatility clustering, QLIKE is a more suitable evaluation metric than MSE; it is strictly consistent for conditional variance estimation and was used for hyperparameter selection in LSTM and SVR


## Model Experimentation
- GARCH
  - A GARCH(1,1) model with Student’s t innovations was refit daily using a 250-day rolling window
  - Parameters were estimated via maximum likelihood on each window
  - The (1,1) specification (one lag of squared returns and one lag of conditional variance) captures volatility clustering with minimal complexity, justified by persistent autocorrelation in squared returns
  - In the observed timeline, GARCH tracks medium-range volatility regimes well, but underreacts in periods of abrupt volatility reversal
 

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
  - SVR captures nonlines patterns in volatility without sequential memory, yielding a flexible, nonparametric alternative to GARCH
  - In out observed period, SVR closely tracks both smooth and volatile periods, including the mid-2024 spike, smoothing out fluctuations moreso than LSTM

- Hybrid Model
  - A convex combination of the GARCH, LSTM, and SVR forecasts was used to construct a hybrid volatility forecast
  - Weights were optimized under non-negativity and sum-to-one constraints using scipy.optimize.minimize
    - Optimized weights: GARCH = 0.0622, LSTM = 0.4666, SVR = 0.4712
  - LSTM and SVR capture nonlinear persistence, while GARCH contributes regime-level structure.
  - In the observed period, the hybrid forecast adapts well across volatility phases, balancing smoothness with reactivity around the mid-2024 spike

## Results
- Models were assessed using four metrics on the test set: QLIKE (loss-based), MSE (pointwise error), Pearson correlation (tracking alignment), and directional accuracy (sign of change)
  - QLIKE: Hybrid yields the lowest QLIKE (−3.32), outperforming SVR and LSTM. GARCH’s higher loss (−3.25) reflects poor adaptation to regime changes
  - MSE: SVR and Hybrid minimize MSE (0.000063), indicating sharper pointwise accuracy. GARCH’s elevated error signals poor short-term precision.
  - Correlation: SVR and Hybrid show strongest correlation (≈0.973) with realized volatility, closely matching overall dynamics. GARCH underperforms (≈0.727), missing structural patterns.
  - GARCH and SVR (≈0.543) achieve the highest directional accuracy. Hybrid is slightly lower (~0.532), while LSTM underperforms (~0.526), likely due to delayed response to volatility shifts.
- SVR and LSTM outperform GARCH by capturing nonlinear and sequential volatility patterns, while the Hybrid model combines their strengths for more robust, regime-adaptive forecasts.
