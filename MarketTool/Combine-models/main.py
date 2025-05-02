import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import random
import joblib
from datetime import datetime, timedelta

# Import directly from your existing modules
from data_prep import download_stock_data, get_sp500_tickers
from indicators import generate_features
from trade_tracking import TradeManager
from filters import detect_market_regime
import mass_backtesting

# Set page configuration
st.set_page_config(
    page_title="MarketTool ML Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Function to generate dummy data when downloads fail
def generate_dummy_data(ticker, start_date, end_date):
    """Generate dummy data for testing when downloads fail"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start, end=end, freq='B')
    
    close_price = 100
    prices = []
    for _ in range(len(date_range)):
        close_price *= (1 + random.uniform(-0.02, 0.02))
        open_price = close_price * (1 + random.uniform(-0.01, 0.01))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        prices.append([open_price, high_price, low_price, close_price, int(close_price * 1000)])
    
    df = pd.DataFrame(
        prices, 
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=date_range
    )
    
    st.warning(f"Using simulated data for {ticker} (download failed)")
    return df

# Wrapper function to run backtest using mass_backtesting's implementation
def run_backtest(initial_capital, selected_tickers, start_date, end_date,
                stop_loss_pct, profit_target_pct, trailing_stop_pct, 
                max_duration_days, slippage_pct, use_simulated_data=False):
    """
    Wrapper function that uses the actual mass_backtesting implementation
    """
    try:
        # Progress tracking for UI
        progress_placeholder = st.empty()
        progress_placeholder.text(f"Setting up backtest...")
        
        # Define models path for mass_backtesting
        # The mass_backtesting module might be looking in a specific location
        model_paths = [
            "models",
            "MarketTool/Combine-models/models",
            "../models"
        ]
        
        # Ensure the models directory is in the path 
        # (mass_backtesting might be hardcoded to look in a specific place)
        for path in model_paths:
            if os.path.exists(path):
                mass_backtesting.model_path = path
                break
        
        # Check if we need to handle simulated data
        if use_simulated_data:
            # Create a temporary function to override download_stock_data
            original_download = mass_backtesting.download_stock_data
            
            def simulated_download(ticker, start, end, **kwargs):
                return generate_dummy_data(ticker, start, end)
            
            # Swap the functions
            mass_backtesting.download_stock_data = simulated_download
        
        # Show progress updates
        progress_placeholder.text(f"Running backtest on {len(selected_tickers)} stocks...")
        
        # Add this before calling run_enhanced_backtest:
        import inspect
        st.write("Available parameters for run_enhanced_backtest:", 
                 inspect.signature(mass_backtesting.run_enhanced_backtest))
        
        # Call the ACTUAL implementation from mass_backtesting
        # This ensures we're using the same strategy that works well in terminal
        result = mass_backtesting.run_enhanced_backtest(
            initial_portfolio=initial_capital,
            selected_tickers=selected_tickers,  # Changed from "tickers" to "selected_tickers"
            start_date=start_date,
            end_date=end_date,
            stop_loss_pct=stop_loss_pct,
            profit_target_pct=profit_target_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_duration_days=max_duration_days,
            slippage_pct=slippage_pct
        )
        
        # Restore original download function if we overrode it
        if use_simulated_data:
            mass_backtesting.download_stock_data = original_download
        
        # Clean up progress indicator
        progress_placeholder.empty()
        
        # Extract results - may need to adjust based on actual return values
        if isinstance(result, tuple):
            # If run_enhanced_backtest returns a tuple, unpack it properly
            if len(result) >= 3:  # If it returns (trade_manager, performance_metrics, ticker_performance)
                trade_manager, performance_metrics, ticker_performance = result
                return trade_manager, ticker_performance
            elif len(result) >= 2:  # If it returns (trade_manager, ticker_performance)
                trade_manager, ticker_performance = result
                return trade_manager, ticker_performance
            else:  # If it returns just trade_manager
                trade_manager = result[0]
                return trade_manager, {}
        else:
            # If it returns just the trade_manager
            trade_manager = result
            return trade_manager, {}
        
    except Exception as e:
        st.error(f"Error in backtest: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# Function to create benchmark comparison
def create_benchmark_comparison(trade_manager, selected_tickers, start_date, end_date):
    try:
        # Use mass_backtesting's comparison function if it exists
        if hasattr(mass_backtesting, 'compare_strategy_to_benchmark'):
            benchmark_data = mass_backtesting.compare_strategy_to_benchmark(
                trade_manager=trade_manager,
                tickers=selected_tickers,
                start_date=start_date,
                end_date=end_date
            )
            return benchmark_data
    except Exception as e:
        st.warning(f"Could not use built-in benchmark comparison: {str(e)}")
    
    # Fallback to our implementation if the above fails
    try:
        # Download SPY as benchmark
        spy_data = download_stock_data('SPY', start=start_date, end=end_date)
        if spy_data.empty:
            spy_data = generate_dummy_data('SPY', start_date, end_date)
        
        spy_data = spy_data['Close'].to_frame().rename(columns={'Close': 'SPY'})
        benchmark_df = spy_data
        
        # Add individual ticker performance
        for ticker in selected_tickers:
            try:
                ticker_data = download_stock_data(ticker, start=start_date, end=end_date)
                if ticker_data.empty:
                    ticker_data = generate_dummy_data(ticker, start_date, end_date)
                
                if 'Close' in ticker_data.columns:
                    benchmark_df[ticker] = ticker_data['Close']
                    time.sleep(0.5)  # Small delay to avoid API limits
            except Exception as e:
                st.warning(f"Couldn't add {ticker} to benchmark: {str(e)}")
                
        return benchmark_df
    except Exception as e:
        st.warning(f"Couldn't create benchmark comparison: {str(e)}")
        return None

# Main app
def main():
    # Title and introduction
    st.title("📊 ML Trading Strategy Backtester")
    st.markdown("""
    This dashboard backtests our sophisticated ML trading strategy on historical stock data.
    Configure your portfolio, risk parameters, and see how the models perform against market benchmarks.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("📝 Configuration")
    
    # Portfolio settings
    st.sidebar.subheader("Portfolio")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000.00, step=1000.0, format="%.2f")
    
    # Stock selection
    st.sidebar.subheader("Stock Selection")
    stock_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"] + get_sp500_tickers()[:20]
    selected_tickers = st.sidebar.multiselect("Select Stocks", options=stock_options, default=["AAPL"])
    
    # Date range
    st.sidebar.subheader("Backtest Period")
    today = datetime.now()
    prev_year = today - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", value=prev_year)
    end_date = st.sidebar.date_input("End Date", value=today)
    
    # Risk Management
    st.sidebar.subheader("Risk Management")
    st.sidebar.caption("Press Enter for defaults")
    stop_loss_pct = st.sidebar.number_input("Stop Loss Percentage", value=0.05, format="%.2f", 
                                            help="Percentage of position value to set stop loss")
    profit_target_pct = st.sidebar.number_input("Profit Target Percentage", value=0.10, format="%.2f", 
                                               help="Percentage of position value to set profit target")
    trailing_stop_pct = st.sidebar.number_input("Trailing Stop Percentage", value=0.03, format="%.2f", 
                                              help="Percentage for trailing stops")
    max_duration_days = st.sidebar.number_input("Maximum Trade Duration (days)", value=30, step=1, 
                                              help="Maximum days to hold a position")
    slippage_pct = st.sidebar.number_input("Slippage Percentage", value=0.001, format="%.3f", 
                                         help="Slippage for trade execution")
    
    # Data source option
    use_simulated_data = st.sidebar.checkbox("Use simulated data (avoid API limits)", value=False)
    use_only_simulated = st.sidebar.checkbox("Use only simulated data (avoid all API calls)", value=False)

    # Run backtest button
    if st.sidebar.button("Run Backtest"):
        if not selected_tickers:
            st.error("Please select at least one stock")
            return
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Setting up backtest...")
        progress_bar.progress(10)
        
        # Run backtest using the mass_backtesting implementation
        status_text.text("Running backtest...")
        trade_manager, ticker_performance = run_backtest(
            initial_capital=initial_capital,
            selected_tickers=selected_tickers,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            stop_loss_pct=stop_loss_pct,
            profit_target_pct=profit_target_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_duration_days=max_duration_days,
            slippage_pct=slippage_pct,
            use_simulated_data=use_only_simulated  # Pass the toggle value
        )
        progress_bar.progress(70)
        
        if trade_manager is None:
            st.error("Backtest failed. Please check logs.")
            return
            
        # Create benchmark comparison
        status_text.text("Creating benchmark comparison...")
        benchmark_df = create_benchmark_comparison(
            trade_manager=trade_manager, 
            selected_tickers=selected_tickers,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        progress_bar.progress(90)
        
        # Get performance metrics from trade manager
        performance_metrics = trade_manager.get_performance_metrics()
        progress_bar.progress(100)
        status_text.empty()
        
        # Display results in main area
        col1, col2 = st.columns(2)
        
        # Portfolio performance visualization
        with col1:
            st.subheader("Portfolio Performance")
            portfolio_value_df = trade_manager.get_portfolio_value_df()
            
            if portfolio_value_df is not None and not portfolio_value_df.empty:
                # Safely find portfolio value column
                value_cols = ['Total_Value', 'portfolio_value', 'total_value', 'value', 'equity', 'balance', 'nav']
                value_col = None
                
                for col in value_cols:
                    if col in portfolio_value_df.columns:
                        value_col = col
                        break
                
                # If no standard column found, try to find any numeric column
                if value_col is None:
                    numeric_cols = portfolio_value_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        value_col = numeric_cols[0]
                        st.info(f"Using column '{value_col}' for portfolio value")
                
                # Display portfolio chart and metrics
                if value_col:
                    st.line_chart(portfolio_value_df[value_col])
                    
                    initial_value = portfolio_value_df[value_col].iloc[0]
                    final_value = portfolio_value_df[value_col].iloc[-1]
                    total_return = ((final_value / initial_value) - 1) * 100
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Initial Value", f"${initial_value:.2f}")
                    metrics_col2.metric("Final Value", f"${final_value:.2f}")
                    metrics_col3.metric("Total Return", f"{total_return:.2f}%")
                else:
                    st.warning("Could not determine portfolio value column")
            else:
                st.warning("No portfolio data available")
        
        # Benchmark comparison
        with col2:
            st.subheader("Benchmark Comparison")
            if benchmark_df is not None and not benchmark_df.empty and value_col:
                # Normalize benchmark data
                normalized_benchmark = benchmark_df.div(benchmark_df.iloc[0]).mul(initial_capital)
                
                # Add strategy performance
                if portfolio_value_df is not None and not portfolio_value_df.empty:
                    normalized_benchmark['Strategy'] = portfolio_value_df[value_col]
                
                st.line_chart(normalized_benchmark)
            else:
                st.warning("Benchmark data not available")
        
        # Trade analysis section
        st.subheader("Trade Analysis")
        col1, col2 = st.columns(2)
        
        # Trade performance table
        with col1:
            st.markdown("### Trade Performance")
            closed_trades_df = trade_manager.get_closed_trades_df()
            
            if closed_trades_df is not None and not closed_trades_df.empty:
                st.dataframe(closed_trades_df)
                
                # Find the return percentage column - being flexible with naming
                return_col = None
                possible_return_cols = [
                    'Return_Pct', 'return_pct', 'return', 'profit_pct', 
                    'pnl_pct', 'pct_return', 'roi'
                ]
                
                for col in possible_return_cols:
                    if col in closed_trades_df.columns:
                        return_col = col
                        break
                
                # Calculate returns if needed
                if return_col is None and 'entry_price' in closed_trades_df.columns and 'exit_price' in closed_trades_df.columns:
                    closed_trades_df['calculated_return'] = (
                        closed_trades_df['exit_price'] / closed_trades_df['entry_price'] - 1
                    )
                    return_col = 'calculated_return'
                
                # Display trade statistics
                if return_col:
                    total_trades = len(closed_trades_df)
                    winning_trades = len(closed_trades_df[closed_trades_df[return_col] > 0])
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    stats_col1.metric("Total Trades", f"{total_trades}")
                    stats_col2.metric("Win Rate", f"{win_rate:.2%}")
                    stats_col3.metric("Avg Return/Trade", f"{closed_trades_df[return_col].mean():.2%}")
                else:
                    total_trades = len(closed_trades_df)
                    stats_col1 = st.columns(1)[0]
                    stats_col1.metric("Total Trades", f"{total_trades}")
            else:
                st.warning("No closed trades in this period")
        
        # Trade distribution visualization
        with col2:
            st.markdown("### Trade Distribution")
            if closed_trades_df is not None and not closed_trades_df.empty and return_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(closed_trades_df[return_col] * 100, bins=20, ax=ax)
                ax.set_xlabel('Return %')
                ax.set_ylabel('Number of Trades')
                ax.set_title('Distribution of Trade Returns')
                st.pyplot(fig)
            else:
                st.warning("No trade return data to visualize")
        
        # Performance metrics section
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        
        # Display performance metrics table
        with col1:
            if performance_metrics:
                metrics_df = pd.DataFrame({
                    'Metric': performance_metrics.keys(),
                    'Value': performance_metrics.values()
                })
                st.dataframe(metrics_df)
            else:
                st.warning("No performance metrics available")
        
        # Monthly returns chart
        with col2:
            if portfolio_value_df is not None and not portfolio_value_df.empty and value_col:
                try:
                    monthly_returns = portfolio_value_df[value_col].resample('M').last().pct_change().dropna()
                    if not monthly_returns.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        monthly_returns.plot(kind='bar', ax=ax)
                        ax.set_title('Monthly Returns')
                        ax.set_ylabel('Return %')
                        ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly_returns.index], rotation=45)
                        st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not calculate monthly returns: {e}")

if __name__ == "__main__":
    main()