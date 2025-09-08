# Stock-price-direction-prediction-using-XGBoost-and-Deep-Learning
Overview:
This repository documents my end-to-end project in predicting short-term stock price movements using high-frequency limit order book (LOB) data from LOBSTER (https://lobsterdata.com/info/DataSamples.php), specifically the Amazon stock level 10 data on the linked webpage.
The project progressed through four stages, starting with tree-based models and evolving into deep learning architectures, culminating in paper-faithful implementations of DeepLOB (Zhang et al.) and Tsantekidis et al.’s CNN-LSTM.

The journey highlights both experimentation and refinement — beginning with my own CNN/LSTM architectures, and then moving toward reproducing academic models.

Project Progression

(Note- the specified files need to be downloaded and unzipped in the same repo as the data above for the code to run)

1. XGBoost Baseline – Feature Engineering + Tree Models

Notebook: XGBOOST classification.ipynb

XGBOOST classification

Engineered financial microstructure features:

Spread, depth ratio, order book imbalance
Rolling trade imbalance, cancel-to-add ratio
Aggressive volume ratio, rolling mid-price change
Rolling volatility, message rate, etc.

Multiclass classification (up, down, neutral).

Models:
Initial XGBoost classifier.
Advanced tuning with Optuna.

Key learning: XGBoost was interpretable and moderately effective, but struggled to capture temporal dependencies.

2. Custom CNN – Raw LOB Data

Notebook: Price direction prediction with CNN.ipynb

Designed my own CNN architecture to directly process sequences of raw LOB states.
Input: 50-timestep sliding windows of level-10 order book (40 features per timestep).
Output: Binary/multiclass classification.

Key learning: CNNs captured local spatial patterns, but performance plateaued without temporal modeling.

3. Custom CNN-LSTM – Adding Temporal Dynamics

Notebook: Price direction prediction with CNN and LSTM.ipynb
Hybrid CNN → LSTM pipeline.
CNN layers extracted spatial/structural features from the LOB.
LSTM captured temporal dependencies across sequences.

Key learning: Adding LSTM improved sequential awareness, but the architecture was hand-crafted, not optimized based on prior research.

4. Paper-Faithful Replication – DeepLOB & Tsantekidis CNN-LSTM

Notebook: Price direction prediction based on academic papers.ipynb
Implemented research-accurate models:
DeepLOB (Zhang et al., 2018): Inception-style CNN + BiLSTM + attention.
Tsantekidis CNN-LSTM (2017/2020): Stationary engineered features + Conv1D + BiLSTM + attention.
Adopted paper-faithful event-based labeling (3-class: up, down, neutral).
Training with 200 epochs, AdamW optimizer, cyclic learning rate.

Key learning: Performance was limited by dataset size (single day of AMZN). Overfitting was expected — but architectures successfully reproduced, creating a scalable pipeline for larger datasets which is the next step in the project.

Data

Source: LOBSTER (sample files).

Instrument: AMZN, June 21, 2012 (9:30–16:00 EST).

Granularity: Level-10 LOB snapshots, ~270k events in one trading day.

Preprocessing:
Raw order book states.
Engineered stationary features (as in Tsantekidis).
Event-based labeling (future mid-price returns over horizon H, with tolerance α).

Results (Summary)

XGBoost: Solid baseline, interpretable features, 63% accuracy.

Custom CNN: Captured spatial patterns but struggled with sequence prediction.

Custom CNN-LSTM: Added temporal structure, moderate gains.

DeepLOB / Tsantekidis Replication: Paper-faithful models built, but accuracy/F1 remained low due to limited dataset size.

Key Takeaways

Built full pipeline: data prep → feature engineering → model design → evaluation.

Progressed from classical ML → custom DL → research replication.

Learned the importance of event-based labeling, class balance, and dataset scale.

Code is modular: easy to toggle between raw features vs engineered features, binary vs multiclass setups.

Future Work

Train on multi-day / multi-stock LOBSTER datasets (as in FI-2010, DeepLOB).

Possibly extend beyond CNN-LSTM to other models, like transformers and graph-based models.

Backtest trading strategies using predicted signals.

References

Zhang, Z., Zohren, S., & Roberts, S. (2018). DeepLOB: Deep convolutional neural networks for limit order books. IEEE TSP.

Tsantekidis, A., Passalis, N., et al. (2017). Forecasting stock prices from the limit order book using CNNs and LSTMs.

LOBSTER data: https://lobsterdata.com
