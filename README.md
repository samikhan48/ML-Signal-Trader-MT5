# ML Signal Trader V5 (MT5 / MQL5 / ONNX)

An advanced, automated algorithmic trading pipeline integrating Machine Learning models directly into MetaTrader 5 (MT5) for Forex and commodity markets.

## 🚀 Overview
The ML Signal Trader V5 bridges the gap between Python-based machine learning and real-time execution in MQL5. Instead of relying on slow API bridges during live trading, this project exports trained models (CNN, SVM) into **ONNX** format. The MetaTrader 5 Expert Advisor (EA) directly loads the ONNX models to produce lightning-fast, probability-based execution signals.

## ✨ Key Features
- **ONNX Model Integration:** Native MQL5 execution of Deep Learning algorithms, predicting price action with sub-millisecond latency.
- **Multi-Asset Compatibility:** Backtested and optimized across major FX pairs (`GBPUSD`, `EURUSD`, `AUDUSD`, `USDJPY`) and commodities (`XAUUSD`).
- **Triple-Barrier Method:** Advanced labeling technique used in Python data preparation to account for stop-loss, take-profit, and time-horizons.
- **FinBERT Sentiment Analysis:** Incorporates macroeconomic sentiment pipelines (`sentiment_analyzer.py`) to align technical entries with fundamental biases.
- **Prop-Firm Safe News Filter:** Built-in MQL5 Economic Calendar API queries prevent trade execution around high-volatility news events to avoid slippage and adhere to funding firm rules.

## 🗂️ Core Project Structure
*Note: Due to file sizes, raw tick data and compiled `.ex5` files are omitted from this repository.*

- `ML_Signal_Trader_V5.mq5`: The core Expert Advisor (EA) source code executed in MT5.
- `colab_notebook_v21_onnx_v5_triple_barrier.py`: Python research environment detailing data engineering, Triple Barrier labeling, and ONNX exporting.
- `train_cnn_model.py` / `train_hmm_models.py`: Custom scripts for training Convolutional Neural Networks and Hidden Markov Models (Regime Detection).
- `sentiment_analyzer.py`: Utilizes the NLP model FinBERT to digest and quantify market sentiment.

## ⚙️ How it Works
1. **Data Engineering:** Historical tick and H1 data are pulled via the MT5 python integration. 
2. **Training & Export:** Models are trained using Scikit-Learn/PyTorch and exported to the universally compatible `.onnx` format.
3. **MQL5 Execution:** The EA (`.mq5`) loads the ONNX file dynamically. On each new bar or tick, it feeds the model the latest technical indicators and receives a confidence tensor for `Buy`, `Sell`, or `Hold`.
4. **Risk Management:** The execution is gated by the MQL5 Calendar filter and dynamically sized based on account equity.

## 📈 Tools & Tech Stack
- **Python:** `pandas`, `numpy`, `scikit-learn`, `PyTorch`
- **NLP / Sentiment:** `transformes` (HuggingFace FinBERT)
- **Execution & MQL:** MQL5, MetaTrader 5, ONNX

---
*Disclaimer: This repository is for demonstration and educational purposes to highlight quantitative development skills. It does not constitute financial advice.*
