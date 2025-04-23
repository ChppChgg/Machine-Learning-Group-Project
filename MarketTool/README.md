# Advanced Stock Trading Backtesting System

This repository contains a comprehensive machine learning-based stock trading system that generates predictions, manages trades, and provides detailed performance analysis. The system combines multiple ML models with technical indicators, market regime filters, and sophisticated trade management to create a complete trading solution.

## Table of Contents
- System Overview
- Key Components
- Data Preparation
- Feature Engineering
- Machine Learning Models
- Trade Management
- Market Regime Filtering
- Backtesting Engine
- Performance Visualization
- Usage Guide
- Configuration

## System Overview

This backtesting framework allows traders and researchers to test trading strategies powered by machine learning. The system:

1. Downloads historical stock data
2. Generates technical indicators
3. Uses trained ML models to predict price movements
4. Applies market regime filters to improve signal quality
5. Manages trades with sophisticated entry/exit rules
6. Tracks portfolio performance
7. Visualizes results compared to benchmarks

## Key Components

### Data Preparation (`data_prep.py`)

Handles the acquisition and preparation of financial data:

- **`download_stock_data()`**: Downloads historical price data from Yahoo Finance
- **`create_labels()`**: Creates training labels based on future price movements
- **`get_sp500_tickers()`**: Retrieves current S&P 500 component stocks

### Feature Engineering (`indicators.py`)

Calculates technical indicators that serve as features for the ML models:

- **`calculate_rsi()`**: Relative Strength Index
- **`calculate_macd()`**: Moving Average Convergence Divergence
- **`calculate_bollinger_bands()`**: Bollinger Bands
- **`calculate_atr()`**: Average True Range
- **`calculate_obv()`**: On-Balance Volume
- **`generate_features()`**: Main function that creates all features for model training

### Market Regime Filtering (`filters.py`)

Detects market conditions to improve signal quality:

- **`detect_market_regime()`**: Identifies bullish, bearish, or neutral market conditions
- Calculates trend strength and direction to filter trading signals

### Trade Management (`trade_tracking.py`)

Sophisticated trade management with various exit strategies:

- **`Trade`** class: 
  - Manages individual trade lifecycle
  - Implements stop loss, profit target, trailing stops
  - Tracks trade performance metrics

- **`TradeManager`** class:
  - Manages overall portfolio
  - Tracks all open/closed trades
  - Calculates performance metrics
  - Records daily portfolio values
  - Generates performance reports

- Exit strategy utilities:
  - **`calculate_trailing_stop()`**: Implements trailing stop loss
  - **`volatility_adjusted_stop_loss()`**: Dynamic stop based on volatility
  - **`chandelier_exit()`**: ATR-based trailing stop method

### Machine Learning (`train_models.py`)

Creates and trains prediction models:

- Uses a stacked classifier approach with:
  - Logistic Regression
  - XGBoost
  - Neural Network
- Handles class imbalance with SMOTE
- Saves trained models for later use in predictions

### Backtesting Engine (`mass-backtesting.py`)

Main system that ties everything together:

- **`run_enhanced_backtest()`**: Core backtesting function that:
  - Processes historical data
  - Generates trading signals
  - Manages trades
  - Tracks performance

- **`filter_trading_signals()`**: Improves signal quality by considering:
  - Market regime
  - Technical confirmations
  - Recent performance

- **`compare_strategy_to_benchmark()`**: Compares strategy to buy-and-hold

### Performance Visualization (multiple files)

Generates detailed performance reports and visualizations:

- Portfolio equity curves
- Drawdown analysis
- Trade statistics
- Performance comparison with benchmarks
- Win/loss distribution by exit reason

## Usage Guide

To run a backtest:

1. Execute the mass-backtesting.py file:
   ```
   python mass-backtesting.py
   ```

2. Follow the interactive prompts to:
   - Set initial portfolio value
   - Select test mode (single stock, multiple stocks, random stocks)
   - Choose risk management parameters:
     - Stop loss percentage
     - Profit target percentage
     - Trailing stop percentage
     - Maximum trade duration

3. Review the performance results displayed in the console and saved to the `test-results` directory.

## Configuration

### Risk Management Parameters

- **Stop Loss**: Default 5% (recommended 8-10% for less frequent exits)
- **Profit Target**: Default 10% (recommended 15-20% for better risk/reward)
- **Trailing Stop**: Default 3% (set to 0 to disable)
- **Maximum Trade Duration**: Default 30 days (set to 0 for no limit)

### Market Filtering Options

The system uses trend detection and strength filtering to improve signal quality. These parameters can be adjusted in the `filters.py` and mass-backtesting.py files.

### Performance Metrics

The system tracks and reports:

- Total return
- Annualized return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Average return per trade
- Profit factor
- Average trade duration
- Benchmark comparison

## Example Output

The system generates both console output and saved files including:
- CSV files with trade history
- Portfolio value over time
- Performance metrics
- PNG charts showing portfolio performance
- Trade analysis visualizations
- Strategy vs benchmark comparisons

All results are saved with incremental numbering to prevent overwriting previous results.