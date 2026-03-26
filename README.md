# 🏛️ The Quant Titan (V6.0) - Institutional MT5 Python Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![MetaTrader 5](https://img.shields.io/badge/MetaTrader_5-API-black?logo=alpinelinux)
![Polars](https://img.shields.io/badge/Polars-Data_Engine-orange?logo=polars)
![Asyncio](https://img.shields.io/badge/Architecture-Asyncio-success)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)

## 📌 Overview
**The Quant Titan V6.0** is an enterprise-grade, fully asynchronous algorithmic trading infrastructure designed to bridge Python with MetaTrader 5 (MT5). 

It was built specifically to solve the biggest bottleneck in retail MT5 API development: **Terminal Freezing and Blocking I/O during high-frequency tick polling.** By utilizing a pure `asyncio.to_thread` architecture paired with a blazing-fast `Polars` data engine, this framework allows for microsecond-level feature engineering and execution without ever dropping a tick or freezing the MT5 terminal.

---

## 🚀 Core Architecture & Features

### 1. ⚡ True Asynchronous Execution (`main.py` & `execution.py`)
Standard MT5 Python integrations use blocking calls, which crash the terminal when polling ticks across multiple pairs. 
- **The Solution:** A triple-loop concurrent architecture.
  - **Tick Loop (1.0s):** Polls tick data and manages tick-driven trailing stops.
  - **Strategy Loop (15.0s):** Evaluates the 10-Brain consensus and executes trades.
  - **Candle Loop (60.0s):** Refreshes M15/H1 data and computes heavy indicators.
- All MT5 API calls are wrapped safely to ensure zero GUI lag.

### 2. 🧠 The 10-Brain Ensemble & ML Meta-Brain (`strategy.py`)
Instead of relying on a single flawed indicator, the engine uses a multi-model consensus system.
- **Independent Brains:** Includes Z-Score Anomaly, Hurst Exponent, Smart Money Concepts (SMC), Order Flow / Whale Tracking, Volatility Regimes, Momentum Divergence, and more.
- **Meta-Brain (Machine Learning):** Uses `SGDClassifier` for online learning. It tracks the historical win-rate of each brain and dynamically adjusts their voting weights using Exponential Weighting (EWA).

### 3. 📊 Polars-Powered Data Engine (`data_engine.py`)
Pandas is too slow for tick-level analysis. This engine uses `Polars` and `NumPy`.
- **Microsecond Feature Engineering:** Calculates RSI, MACD, Bollinger Bands, ATR, and Volume Imbalances exponentially faster than Pandas.
- **Zero-NaN Propagation:** Built-in safety protocols ensure that missing MT5 data never propagates into the neural logic.

### 4. 🐋 Institutional Order Flow (`whale_tracker.py`)
- Detects tick-level volume spikes (3x average = Whale).
- Identifies iceberg order clusters and absorption anomalies.
- Polars-based candle volume imbalance analysis.

### 5. 🛡️ Fractional Kelly Risk Management
- Position sizing is not static. It is dynamically calculated using the **Fractional Kelly Criterion**, adjusting the lot size based on the current market regime (Low Volatility vs. High Volatility) and historical edge.

---

## 🧪 Testing & Reliability
This infrastructure is fully unit-tested to ensure zero runtime terminal crashes. 
Check the `test_v6.py` file included in this repository to see the 98+ automated tests covering Risk Management, Brain execution, and Meta-Brain state saving.

---

## 💼 Acquisition / Source Code
*Note: This repository only contains the Architecture Overview and the Test Coverage File to demonstrate the system's capabilities.*

**The complete source code (including Strategy, Execution, Data Engine, and Main Orchestrator) is available for private acquisition / Developer Licensing.**

If you are a Prop Firm, a Hedge Fund, or a Freelance Developer looking to save 3-6 months of R&D and avoid the nightmare of MT5 freezing bugs, contact me to acquire the production-ready source code.
