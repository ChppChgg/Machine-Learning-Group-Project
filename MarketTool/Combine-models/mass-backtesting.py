import os
import random
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from data_prep import download_stock_data, get_sp500_tickers
from indicators import generate_features
from trade_tracking import TradeManager
from datetime import datetime
from filters import detect_market_regime

# --- Load models ---
model_path = "MarketTool/Combine-models/models/"
model = joblib.load(os.path.join(model_path, "stacked_classifier.pkl"))
scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

# Add this function at the top of your file, after imports
def get_numbered_filepath(base_path, extension):
    """
    Generate a numbered filename to prevent overwriting existing files
    
    Parameters:
    -----------
    base_path : str
        Base path without extension (e.g., "MarketTool/Combine-models/test-results/CSV/trade_history")
    extension : str
        File extension (e.g., "csv" or "png")
        
    Returns:
    --------
    str
        Path with number appended (e.g., "MarketTool/Combine-models/test-results/CSV/trade_history_1.csv")
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    # Find the next available number
    i = 1
    while os.path.exists(f"{base_path}_{i}.{extension}"):
        i += 1
    
    return f"{base_path}_{i}.{extension}"

# Add this new function
def balance_trading_signals(features, window=5):
    """
    Balance trading signals to avoid one-sided trading
    
    Parameters:
    -----------
    features : pandas.DataFrame
        DataFrame with 'Model_Prediction' column
    window : int
        Rolling window for technical confirmation
    
    Returns:
    --------
    pandas.Series
        Balanced trading signals
    """
    df = features.copy()
    
    # Add simple technical indicators for confirmation
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    
    # Create trend indicator
    df['Trend'] = np.where(df['SMA5'] > df['SMA20'], 'UP', 'DOWN')
    
    # Balance signals using technical confirmation
    balanced_signals = []
    
    for i, row in df.iterrows():
        base_signal = row['Model_Prediction']
        
        # If model says SELL but trend is UP, consider HOLD
        if base_signal == 'SELL' and row['Trend'] == 'UP' and row['Close'] > row['SMA20'] * 1.02:
            balanced_signals.append('HOLD')
        # If model says BUY but trend is DOWN, consider HOLD
        elif base_signal == 'BUY' and row['Trend'] == 'DOWN' and row['Close'] < row['SMA20'] * 0.98:
            balanced_signals.append('HOLD')
        # Otherwise use model signal
        else:
            balanced_signals.append(base_signal)
    
    return pd.Series(balanced_signals, index=df.index)

# Add to your signal generation function or where you process model predictions

def filter_trading_signals(predictions, market_data):
    """Filter signals based on market conditions"""
    
    filtered_signals = []
    
    for i, pred in enumerate(predictions):
        # Skip if we don't have enough data yet
        if i < 50:
            filtered_signals.append(0)
            continue
        
        # Get market conditions for this day
        regime = market_data['market_regime'].iloc[i]
        volatility = market_data['volatility'].iloc[i]
        
        # Only take trades aligned with the market regime
        if pred == "BUY" and regime > 0:
            filtered_signals.append(1)  # Long signal
        elif pred == "SELL" and regime < 0:
            filtered_signals.append(-1)  # Short signal
        else:
            filtered_signals.append(0)  # No trade
            
        # Add additional volatility filter (optional)
        if volatility > 30:  # Skip during extreme volatility
            filtered_signals[-1] = 0
            
    return filtered_signals

# --- Run enhanced backtesting with trade tracking ---
def run_enhanced_backtest(initial_portfolio, selected_tickers, start_date, end_date, 
                          stop_loss_pct=0.05, profit_target_pct=0.10, 
                          trailing_stop_pct=0.03, max_duration_days=30,
                          slippage_pct=0.000):
    """
    Run enhanced backtest with trade tracking and sophisticated exit strategies
    """
    # Initialize trade manager
    trade_manager = TradeManager(
        initial_capital=initial_portfolio,
        slippage_pct=slippage_pct
    )
    
    ticker_performance = {}
    
    for ticker in selected_tickers:
        try:
            print(f"\n=== Processing {ticker} ===")
            # Download data
            df = download_stock_data(ticker, start_date, end_date)
            if df.empty or len(df) < 100:
                print(f"Insufficient data for {ticker}. Skipping.")
                continue
            
            # Generate features
            features = generate_features(df, ticker).dropna()
            
            # Ensure we have a Copy column available for model predictions
            # This is crucial - we need to make a copy of Close before it gets dropped
            features["Close_Price"] = df.loc[features.index, "Close"]
            
            # Prepare data for model
            X = features.drop(columns=["Label", "Ticker", "Date", "Close_Price"], errors="ignore")
            X_scaled = scaler.transform(X)
            
            # Make predictions
            preds = model.predict(X_scaled)
            pred_labels = encoder.inverse_transform(preds)
            features["Model_Prediction"] = pred_labels
            
            # Print prediction distribution
            unique_preds, pred_counts = np.unique(pred_labels, return_counts=True)
            print(f"\nPrediction distribution: {dict(zip(unique_preds, pred_counts))}")
            
            # Apply balancing to predictions
            features["Close"] = features["Close_Price"]  # Make Close available for technical indicators
            
            # Calculate technical indicators for signal balancing
            features["SMA5"] = features["Close"].rolling(5).mean()
            features["SMA20"] = features["Close"].rolling(20).mean()
            features["Trend"] = np.where(features["SMA5"] > features["SMA20"], "UP", "DOWN")
            
            # Apply market direction awareness
            features["Return_5d"] = features["Close"].pct_change(5)
            features["Market_Strength"] = features["Return_5d"].rolling(10).mean().fillna(0)
            
            # Create balanced signals
            features["Signal"] = features["Model_Prediction"].apply(
                lambda x: 1 if x == "BUY" else -1 if x == "SELL" else 0
            )
            
            # Only allow SHORT in downtrends and LONG in uptrends (with some tolerance)
            features.loc[(features["Signal"] == -1) & (features["Market_Strength"] > 0.01), "Signal"] = 0
            features.loc[(features["Signal"] == 1) & (features["Market_Strength"] < -0.01), "Signal"] = 0
            
            # Print signal distribution after balancing
            print(f"Final signal distribution: {features['Signal'].value_counts().to_dict()}")
            
            # Set up price dictionary for all dates
            price_dict = {date: {ticker: row["Close"]} for date, row in df.loc[features.index].iterrows()}
            
            # Add market regime data
            market_data = detect_market_regime(df)
            
            # Process signals day by day
            previous_signal = 0
            last_exit_date = None
            for date, row in features.iterrows():
                current_signal = row["Signal"]
                current_price = row["Close"]
                
                # Filter signals - only take strong trends
                current_regime = market_data.loc[date]
                trend = current_regime['trend'] 
                strength = current_regime['strength']
                
                if current_signal == 1:  # BUY signal
                    if trend < 0 and strength > 0.3:  # Strong downtrend
                        current_signal = 0  # Don't take the trade
                elif current_signal == -1:  # SELL signal
                    if trend > 0 and strength > 0.3:  # Strong uptrend
                        current_signal = 0  # Don't take the trade
                
                # Add cooldown between trades (optional)
                if last_exit_date is not None and (date - last_exit_date).days < 3:
                    current_signal = 0  # Skip trading during cooldown
                
                # Check if we should open a new trade
                if current_signal != 0 and current_signal != previous_signal:
                    # If signal changed and we have an open trade, close it
                    if ticker in trade_manager.open_trades:
                        trade_manager.force_close_trade(ticker, date, current_price)
                        last_exit_date = date
                    
                    # Open a new trade if signal is BUY or SELL
                    if current_signal in [1, -1]:
                        position_type = "LONG" if current_signal == 1 else "SHORT"
                        # Calculate position size (equal allocation per stock)
                        position_size = min(
                            trade_manager.available_capital,
                            initial_portfolio / len(selected_tickers) * 0.95  # Use 95% of allocation
                        )
                        
                        # Skip if position size is too small
                        if position_size < 100:  # Minimum $100 per trade
                            continue
                            
                        trade_manager.open_trade(
                            ticker=ticker,
                            date=date,
                            price=current_price,
                            signal="BUY" if current_signal == 1 else "SELL",
                            position_size=position_size,
                            stop_loss_pct=stop_loss_pct,
                            profit_target_pct=profit_target_pct,
                            trailing_stop_pct=trailing_stop_pct,
                            max_duration_days=max_duration_days
                        )
                
                # Update all trades with today's prices
                trade_manager.update_trades(date, price_dict[date])
                previous_signal = current_signal
            
            # Store ticker performance metrics
            ticker_trades = [t for t in trade_manager.closed_trades if t.ticker == ticker]
            if ticker_trades:
                ticker_returns = [t.calculate_return() for t in ticker_trades]
                win_rate = sum(1 for r in ticker_returns if r > 0) / len(ticker_returns)
                
                ticker_performance[ticker] = {
                    "trades": len(ticker_trades),
                    "win_rate": win_rate * 100,
                    "avg_return": np.mean(ticker_returns) * 100,
                    "max_return": max(ticker_returns) * 100,
                    "min_return": min(ticker_returns) * 100,
                    "avg_duration": np.mean([t.get_trade_summary()["duration_days"] for t in ticker_trades])
                }
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            # For debugging
            import traceback
            traceback.print_exc()
    
    # Close any remaining open trades at the last available price
    for ticker in list(trade_manager.open_trades.keys()):
        trade = trade_manager.open_trades[ticker]
        last_price = trade.current_price  # Use last known price
        trade_manager.force_close_trade(ticker, trade.current_date, last_price)
    
    # Calculate overall performance metrics
    performance_metrics = trade_manager.get_performance_metrics()
    
    return trade_manager, performance_metrics, ticker_performance

# --- User input for portfolio and test mode ---
def get_user_inputs():
    # Get initial portfolio value
    while True:
        try:
            initial_portfolio = float(input("\nEnter initial portfolio value (e.g. 1000): $"))
            if initial_portfolio <= 0:
                print("Portfolio value must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get test mode preference
    print("\nSelect backtesting mode:")
    print("1. Test on a single stock")
    print("2. Test on multiple specific stocks")
    print("3. Test on random stocks")
    
    while True:
        try:
            mode = int(input("Enter your choice (1-3): "))
            if mode not in [1, 2, 3]:
                print("Please enter 1, 2, or 3.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get tickers based on mode selection
    all_tickers = get_sp500_tickers()
    selected_tickers = []
    
    if mode == 1:
        while True:
            ticker = input("Enter a stock ticker symbol (e.g. AAPL): ").upper()
            if ticker in all_tickers:
                selected_tickers = [ticker]
                break
            else:
                print(f"'{ticker}' not found in S&P 500 list. Please try again.")
                
    elif mode == 2:
        print("Enter stock ticker symbols (e.g. AAPL MSFT GOOG), or type 'done' when finished:")
        while True:
            ticker = input("Enter ticker (or 'done'): ").upper()
            if ticker == 'DONE':
                if not selected_tickers:
                    print("You must select at least one stock.")
                    continue
                break
            if ticker in all_tickers:
                selected_tickers.append(ticker)
                print(f"Added {ticker}. Current selection: {', '.join(selected_tickers)}")
            else:
                print(f"'{ticker}' not found in S&P 500 list. Please try again.")
                
    elif mode == 3:
        while True:
            try:
                num_stocks = int(input("Enter number of random stocks to test: "))
                if num_stocks <= 0 or num_stocks > len(all_tickers):
                    print(f"Please enter a number between 1 and {len(all_tickers)}.")
                    continue
                selected_tickers = random.sample(all_tickers, num_stocks)
                print(f"Selected: {', '.join(selected_tickers)}")
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Get date range
    start_date = input("\nEnter start date (YYYY-MM-DD) [default: 2023-01-01]: ")
    if not start_date:
        start_date = "2023-01-01"
    
    end_date = input("Enter end date (YYYY-MM-DD) [default: 2024-12-01]: ")
    if not end_date:
        end_date = "2024-12-01"
    
    # Get risk management parameters
    print("\nRisk Management Settings (press Enter for defaults):")
    stop_loss_pct = float(input("Stop Loss percentage [default: 5%]: ") or 5) / 100
    profit_target_pct = float(input("Profit Target percentage [default: 10%]: ") or 10) / 100
    trailing_stop_pct_input = input("Trailing Stop percentage (0 to disable) [default: 3%]: ")
    trailing_stop_pct = 0 if trailing_stop_pct_input == "0" else float(trailing_stop_pct_input or 3) / 100
    max_duration_input = input("Maximum Trade Duration in days (0 for no limit) [default: 30]: ")
    max_duration_days = None if max_duration_input == "0" else int(max_duration_input or 30)
    slippage_pct = float(input("Slippage percentage [default: 0.1%]: ") or 0.1) / 100
    
    return (initial_portfolio, selected_tickers, start_date, end_date, 
            stop_loss_pct, profit_target_pct, trailing_stop_pct, 
            max_duration_days, slippage_pct)

# --- Visualize detailed results ---
def visualize_enhanced_results(trade_manager, performance_metrics, ticker_performance, selected_tickers, start_date, end_date):
    """
    Visualize enhanced backtest results
    """
    # Create results folders if they don't exist
    os.makedirs("MarketTool/Combine-models/test-results/CSV", exist_ok=True)
    os.makedirs("MarketTool/Combine-models/test-results/PNG", exist_ok=True)
    
    # Save trade history to CSV with numbered filename
    trade_history_path = get_numbered_filepath("MarketTool/Combine-models/test-results/CSV/trade_history", "csv")
    trade_history_df = trade_manager.get_trade_history_df()
    trade_history_df.to_csv(trade_history_path, index=False)
    
    # Save closed trades to CSV with numbered filename
    closed_trades_path = get_numbered_filepath("MarketTool/Combine-models/test-results/CSV/closed_trades", "csv")
    closed_trades_df = trade_manager.get_closed_trades_df()
    closed_trades_df.to_csv(closed_trades_path, index=False)
    
    # Save portfolio value to CSV with numbered filename
    portfolio_path = get_numbered_filepath("MarketTool/Combine-models/test-results/CSV/portfolio_value", "csv")
    portfolio_df = trade_manager.get_portfolio_value_df()
    portfolio_df.to_csv(portfolio_path, index=True)
    
    # Print summary results
    print("\n=== Performance Summary ===")
    print(f"Initial Capital: ${performance_metrics['initial_capital']:.2f}")
    print(f"Final Portfolio Value: ${performance_metrics['final_value']:.2f}")
    print(f"Total Return: {performance_metrics['total_return_pct']:.2f}%")
    print(f"Annualized Return: {performance_metrics['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {performance_metrics['max_drawdown_pct']:.2f}%")
    
    print("\n=== Trading Statistics ===")
    print(f"Total Trades: {performance_metrics['total_trades']}")
    print(f"Win Rate: {performance_metrics['win_rate_pct']:.2f}%")
    print(f"Average Return per Trade: {performance_metrics['avg_return_pct']:.2f}%")
    print(f"Average Winning Trade: {performance_metrics['avg_win_pct']:.2f}%")
    print(f"Average Losing Trade: {performance_metrics['avg_loss_pct']:.2f}%")
    print(f"Profit Factor: {performance_metrics['profit_factor']:.2f}")
    print(f"Average Trade Duration: {performance_metrics['avg_trade_duration']:.1f} days")
    
    print("\n=== Stock-Specific Performance ===")
    ticker_df = pd.DataFrame(ticker_performance).T
    ticker_df.columns = ['# Trades', 'Win Rate %', 'Avg Return %', 'Max Return %', 'Min Return %', 'Avg Duration']
    ticker_df = ticker_df.sort_values('Avg Return %', ascending=False)
    print(ticker_df)
    
    # Save stock performance to CSV with numbered filename
    stock_perf_path = get_numbered_filepath("MarketTool/Combine-models/test-results/CSV/stock_performance", "csv")
    ticker_df.to_csv(stock_perf_path)
    
    # --- Create visualizations with numbered filenames ---
    # 1. Portfolio Performance
    portfolio_perf_path = get_numbered_filepath("MarketTool/Combine-models/test-results/PNG/portfolio_performance", "png")
    trade_manager.plot_portfolio_performance(save_path=portfolio_perf_path)
    
    # 2. Trade Analysis
    if trade_manager.closed_trades:
        trade_analysis_path = get_numbered_filepath("MarketTool/Combine-models/test-results/PNG/trade_analysis", "png")
        trade_manager.plot_trade_analysis(save_path=trade_analysis_path)
    
        
    # Add strategy vs benchmark comparison
    benchmark_metrics = compare_strategy_to_benchmark(trade_manager, selected_tickers, start_date, end_date)
    
    # Update performance metrics with outperformance data
    performance_metrics['benchmark_return_pct'] = benchmark_metrics['benchmark_return'] * 100
    performance_metrics['outperformance_pct'] = benchmark_metrics['outperformance'] * 100
    
    # Print output locations
    print("\nResults saved to:")
    print(f"- Trade history: {trade_history_path}")
    print(f"- Portfolio performance chart: {portfolio_perf_path}")

def compare_strategy_to_benchmark(trade_manager, selected_tickers, start_date, end_date):
    """
    Compare strategy performance to buy-and-hold benchmark and calculate outperformance
    """
    # Get portfolio performance
    portfolio_df = trade_manager.get_portfolio_value_df()
    
    # Check what columns are available in the portfolio DataFrame
    # The column name might be 'Portfolio' or something else, not 'Total'
    if 'Portfolio' in portfolio_df.columns:
        value_column = 'Portfolio'
    else:
        # If no specific column found, use the first numerical column
        numeric_cols = portfolio_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            value_column = numeric_cols[0]
        else:
            # As a fallback, create a new column with the daily portfolio values
            portfolio_df['Strategy'] = trade_manager.portfolio_values_list
            value_column = 'Strategy'
    
    # Calculate strategy return
    portfolio_return = (portfolio_df[value_column].iloc[-1] / portfolio_df[value_column].iloc[0]) - 1
    
    # Calculate buy-and-hold returns for each ticker
    benchmark_returns = {}
    initial_capital = trade_manager.initial_capital
    equal_allocation = initial_capital / len(selected_tickers)
    
    # Create a DataFrame to store benchmark values over time
    benchmark_df = pd.DataFrame(index=portfolio_df.index)
    benchmark_df['Strategy'] = portfolio_df[value_column]  # Use the identified value column
    
    for ticker in selected_tickers:
        try:
            # Download data for the full period
            df = download_stock_data(ticker, start_date, end_date)
            if df.empty:
                continue
                
            # Calculate buy-and-hold return
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            bh_return = (end_price / start_price) - 1
            benchmark_returns[ticker] = bh_return
            
            # Calculate daily buy-and-hold values
            shares = equal_allocation / start_price
            ticker_values = df['Close'] * shares
            
            # Add to benchmark DataFrame, aligning dates
            ticker_values = ticker_values.reindex(portfolio_df.index, method='ffill')
            benchmark_df[f'BH_{ticker}'] = ticker_values
            
        except Exception as e:
            print(f"Error calculating buy-and-hold for {ticker}: {e}")
    
    # Calculate overall benchmark value (equal-weighted portfolio)
    benchmark_cols = [c for c in benchmark_df.columns if c.startswith('BH_')]
    if benchmark_cols:
        benchmark_df['Benchmark'] = benchmark_df[benchmark_cols].sum(axis=1)
        
        # Calculate outperformance
        benchmark_return = (benchmark_df['Benchmark'].iloc[-1] / benchmark_df['Benchmark'].iloc[0]) - 1
        outperformance = portfolio_return - benchmark_return
    else:
        # Handle the case where no valid benchmark data exists
        benchmark_return = 0
        outperformance = 0
        benchmark_df['Benchmark'] = benchmark_df['Strategy']
    
    # Create comparison chart
    plt.figure(figsize=(12, 6))
    
    # Normalize to 100
    strategy_norm = benchmark_df['Strategy'] / benchmark_df['Strategy'].iloc[0] * 100
    benchmark_norm = benchmark_df['Benchmark'] / benchmark_df['Benchmark'].iloc[0] * 100
    
    plt.plot(strategy_norm, label='Trading Strategy', linewidth=2)
    plt.plot(benchmark_norm, label='Buy & Hold', linewidth=2, alpha=0.7)
    
    # Add strategy outperformance annotation
    outperf_str = f"Outperformance: {outperformance*100:.2f}%"
    plt.annotate(outperf_str, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('Strategy Performance vs Buy & Hold (Normalized to 100)')
    plt.xlabel('Date')
    plt.ylabel('Value (Starting = 100)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    comparison_path = get_numbered_filepath("MarketTool/Combine-models/test-results/PNG/strategy_vs_benchmark", "png")
    plt.savefig(comparison_path)
    
    # Save comparison data to CSV
    comparison_data_path = get_numbered_filepath("MarketTool/Combine-models/test-results/CSV/strategy_vs_benchmark", "csv")
    comparison_df = pd.DataFrame({
        'Date': benchmark_df.index,
        'Strategy_Value': benchmark_df['Strategy'],
        'Benchmark_Value': benchmark_df['Benchmark'],
        'Strategy_Normalized': strategy_norm,
        'Benchmark_Normalized': benchmark_norm
    })
    comparison_df.to_csv(comparison_data_path, index=False)
    
    # Save outperformance metrics
    metrics_path = get_numbered_filepath("MarketTool/Combine-models/test-results/CSV/outperformance_metrics", "csv")
    metrics_df = pd.DataFrame({
        'Metric': ['Strategy_Return', 'Benchmark_Return', 'Outperformance'],
        'Value_Pct': [portfolio_return*100, benchmark_return*100, outperformance*100]
    })
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"\n=== Strategy vs Benchmark ===")
    print(f"Strategy Return: {portfolio_return*100:.2f}%")
    print(f"Buy & Hold Return: {benchmark_return*100:.2f}%")
    print(f"Outperformance: {outperformance*100:.2f}%")
    
    print(f"\nComparison charts saved to: {comparison_path}")
    
    return {
        'strategy_return': portfolio_return,
        'benchmark_return': benchmark_return,
        'outperformance': outperformance,
        'comparison_path': comparison_path,
        'metrics_path': metrics_path
    }

# --- Main execution ---
def main():
    print("=== Enhanced Stock Trading Backtest with Trade Management ===")
    
    # Get user inputs including risk management parameters
    (initial_portfolio, selected_tickers, start_date, end_date, 
     stop_loss, profit_target, trailing_stop, max_duration, slippage) = get_user_inputs()
    
    # Run enhanced backtest
    trade_manager, performance_metrics, ticker_performance = run_enhanced_backtest(
        initial_portfolio=initial_portfolio,
        selected_tickers=selected_tickers,
        start_date=start_date,
        end_date=end_date,
        stop_loss_pct=stop_loss,
        profit_target_pct=profit_target,
        trailing_stop_pct=trailing_stop,
        max_duration_days=max_duration,
        slippage_pct=slippage
    )
    
    # Check trade balance
    trade_manager.check_position_balance()
    
    # Visualize results
    if hasattr(trade_manager, 'trade_history') and trade_manager.trade_history:
        visualize_enhanced_results(trade_manager, performance_metrics, ticker_performance, 
                                  selected_tickers, start_date, end_date)
    else:
        print("No trades were made during the backtest period.")
    
if __name__ == "__main__":
    main()