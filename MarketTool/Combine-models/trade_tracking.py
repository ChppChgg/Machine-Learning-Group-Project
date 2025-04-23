import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

class Trade:
    """
    Class to track and manage individual trades
    """
    def __init__(self, ticker, entry_date, entry_price, position_type, position_size, 
                 stop_loss_pct=0.05, profit_target_pct=0.10, trailing_stop_pct=None,
                 max_duration_days=None):
        """
        Initialize a new trade with entry parameters and risk management settings
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        entry_date : datetime
            Entry date of the trade
        entry_price : float
            Entry price of the trade
        position_type : str
            'LONG' or 'SHORT'
        position_size : float
            Dollar amount invested in this trade
        stop_loss_pct : float
            Stop loss as percentage (0.05 = 5%)
        profit_target_pct : float
            Profit target as percentage (0.10 = 10%)
        trailing_stop_pct : float or None
            Trailing stop as percentage, if None, no trailing stop
        max_duration_days : int or None
            Maximum number of days to hold the trade, if None, no time limit
        """
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.position_type = position_type  # 'LONG' or 'SHORT'
        self.initial_position_size = position_size
        self.position_size = position_size  # May change if we implement partial exits
        
        # Current state
        self.current_date = entry_date
        self.current_price = entry_price
        self.highest_price = entry_price  # For trailing stops (long positions)
        self.lowest_price = entry_price   # For trailing stops (short positions)
        self.shares = position_size / entry_price if position_type == 'LONG' else position_size / entry_price
        
        # Risk management parameters
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_duration_days = max_duration_days
        
        # Exit information
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.is_open = True
        
        # Performance tracking
        self.daily_values = {entry_date: position_size}
        self.trade_history = [{
            'date': entry_date,
            'action': 'ENTRY',
            'price': entry_price,
            'position_size': position_size,
            'shares': self.shares
        }]

    def update(self, current_date, current_price):
        """
        Update trade with new price data and check exit conditions
        
        Returns:
        --------
        dict or None
            If trade was exited, returns exit details, else None
        """
        if not self.is_open:
            return None
            
        self.current_date = current_date
        self.current_price = current_price
        
        # Update high/low price points
        if self.position_type == 'LONG':
            self.highest_price = max(self.highest_price, current_price)
        else:  # SHORT
            self.lowest_price = min(self.lowest_price, current_price)
        
        # Calculate current value and store it
        current_value = self._calculate_position_value()
        self.daily_values[current_date] = current_value
        
        # Check exit conditions
        exit_triggered, exit_reason = self._check_exit_conditions()
        if exit_triggered:
            self._exit(current_date, current_price, exit_reason)
            return {
                'ticker': self.ticker,
                'entry_date': self.entry_date,
                'entry_price': self.entry_price,
                'exit_date': self.exit_date,
                'exit_price': self.exit_price,
                'position_type': self.position_type,
                'initial_investment': self.initial_position_size,
                'exit_value': self._calculate_position_value(),
                'return_pct': self.calculate_return(),
                'duration_days': (self.exit_date - self.entry_date).days,
                'exit_reason': self.exit_reason
            }
        return None

    def _calculate_position_value(self, price=None):
        """Calculate current position value"""
        current_price = price if price is not None else self.current_price
        
        if self.position_type == 'LONG':
            return self.shares * current_price
        else:  # SHORT
            # For short positions, we gain when price decreases
            initial_value = self.initial_position_size
            pct_change = 1 - (current_price / self.entry_price)
            return initial_value * (1 + pct_change)
    
    def _check_exit_conditions(self):
        """Check if any exit conditions are met"""
        # Stop Loss and Profit Target checks remain unchanged
        if self.position_type == 'LONG':
            current_return = self.current_price / self.entry_price - 1
            if self.stop_loss_pct is not None and current_return <= -self.stop_loss_pct:
                return True, "STOP_LOSS"
            if self.profit_target_pct is not None and current_return >= self.profit_target_pct:
                return True, "PROFIT_TARGET"
        else:  # SHORT
            current_return = 1 - self.current_price / self.entry_price
            if self.stop_loss_pct is not None and current_return <= -self.stop_loss_pct:
                return True, "STOP_LOSS"
            if self.profit_target_pct is not None and current_return >= self.profit_target_pct:
                return True, "PROFIT_TARGET"
        
        # Enhanced trailing stop logic
        if self.trailing_stop_pct is not None:
            if self.position_type == 'LONG':
                # Only activate trailing stop after price has moved in our favor by at least 1/3 of profit target
                activation_threshold = self.entry_price * (1 + self.profit_target_pct/3)
                if self.highest_price >= activation_threshold:
                    trailing_stop_price = self.highest_price * (1 - self.trailing_stop_pct)
                    if self.current_price <= trailing_stop_price:
                        return True, "TRAILING_STOP"
            else:  # SHORT
                activation_threshold = self.entry_price * (1 - self.profit_target_pct/3)
                if self.lowest_price <= activation_threshold:
                    trailing_stop_price = self.lowest_price * (1 + self.trailing_stop_pct)
                    if self.current_price >= trailing_stop_price:
                        return True, "TRAILING_STOP"
        
        # Check max duration
        if self.max_duration_days is not None:
            duration = (self.current_date - self.entry_date).days
            if duration >= self.max_duration_days:
                return True, "MAX_DURATION"
        
        return False, None
        
    def _exit(self, exit_date, exit_price, reason):
        """Exit the trade and record exit information"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        self.is_open = False
        
        # Record the exit in trade history
        self.trade_history.append({
            'date': exit_date,
            'action': 'EXIT',
            'price': exit_price,
            'position_size': self._calculate_position_value(),
            'shares': self.shares,
            'reason': reason
        })
    
    def calculate_return(self):
        """Calculate the return percentage of this trade"""
        if self.position_type == 'LONG':
            if self.is_open:
                return (self.current_price / self.entry_price) - 1
            else:
                return (self.exit_price / self.entry_price) - 1
        else:  # SHORT
            if self.is_open:
                return 1 - (self.current_price / self.entry_price)
            else:
                return 1 - (self.exit_price / self.entry_price)
    
    def get_trade_summary(self):
        """Get a summary of this trade"""
        return {
            'ticker': self.ticker,
            'position_type': self.position_type,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date if not self.is_open else None,
            'exit_price': self.exit_price if not self.is_open else None,
            'current_price': self.current_price,
            'is_open': self.is_open,
            'initial_investment': self.initial_position_size,
            'current_value': self._calculate_position_value(),
            'return_pct': self.calculate_return() * 100,  # as percentage
            'max_price': self.highest_price if self.position_type == 'LONG' else self.lowest_price,
            'exit_reason': self.exit_reason if not self.is_open else None,
            'duration_days': (self.current_date - self.entry_date).days
        }
        
    def get_daily_values_df(self):
        """Get daily values as DataFrame"""
        df = pd.DataFrame(
            list(self.daily_values.values()), 
            index=list(self.daily_values.keys()),
            columns=['Value']
        )
        df.index.name = 'Date'
        return df

    def update_trailing_stop(self, current_price):
        """Enhanced trailing stop that activates only after profit threshold"""
        
        if self.trailing_stop_pct == 0:
            return False
        
        # For long positions
        if self.position_type == 'LONG':
            # Track highest price seen
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Only activate trailing stop after we've moved 1/3 toward profit target
            activation_threshold = self.entry_price * (1 + self.profit_target_pct/3)
            
            if self.highest_price > activation_threshold:
                # Calculate trailing stop level
                stop_level = self.highest_price * (1 - self.trailing_stop_pct/100)
                
                # Exit if price falls to stop level
                if current_price <= stop_level:
                    return True
        
        # For short positions (similar logic)
        elif self.position_type == 'SHORT':
            # Similar implementation for shorts
            pass
            
        return False
        
class TradeManager:
    """
    Class to manage a portfolio of trades
    """
    def __init__(self, initial_capital, slippage_pct=0.001):
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.open_trades = {}
        self.closed_trades = []
        self.portfolio_history = []  # Daily portfolio value snapshots
        self.daily_portfolio_value = {}  # Add this dictionary to track daily values
        self.trade_history = []  # Add this to track trade history
        self.last_update_date = None
        
    def open_trade(self, ticker, date, price, signal, position_size=None, 
                   stop_loss_pct=0.05, profit_target_pct=0.10,
                   trailing_stop_pct=None, max_duration_days=None):
        """
        Open a new trade position
        """
        if ticker in self.open_trades:
            print(f"Already have an open position in {ticker}. Skipping.")
            return None
            
        # Apply slippage to the execution price
        direction = 1 if signal == 'BUY' else -1
        effective_price = price * (1 + (self.slippage_pct * direction))
        
        # Determine position size
        if position_size is None:
            position_size = self.available_capital * 0.1  # Default to 10% of available capital
        
        position_size = min(position_size, self.available_capital)
        
        if position_size <= 0:
            print("Insufficient capital to open trade.")
            return None
            
        # Calculate number of shares
        shares = position_size / effective_price
        
        position_type = 'LONG' if signal == 'BUY' else 'SHORT'
        
        # Create trade object
        trade = Trade(
            ticker=ticker,
            entry_date=date,
            entry_price=effective_price,
            position_type=position_type,
            position_size=position_size, 
            stop_loss_pct=stop_loss_pct,
            profit_target_pct=profit_target_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_duration_days=max_duration_days
        )
        
        # Store the trade and adjust capital
        self.open_trades[ticker] = trade
        self.available_capital -= position_size
        
        # Record portfolio state
        self._record_portfolio_state(date)
        
        return trade
    
    def force_close_trade(self, ticker, date, price):
        """
        Force close a trade at the given price
        """
        if ticker not in self.open_trades:
            return False
            
        trade = self.open_trades[ticker]
        
        # Apply slippage to the execution price (slippage hurts in the direction of the trade)
        direction = -1 if trade.position_type == 'LONG' else 1
        effective_price = price * (1 + (self.slippage_pct * direction))
        
        # Close trade and return capital 
        exit_value = trade._calculate_position_value(effective_price)
        trade._exit(date, effective_price, "FORCE_CLOSE")
        
        # Add to closed trades and remove from open trades
        self.closed_trades.append(trade)
        del self.open_trades[ticker]
        
        # Return capital to available
        self.available_capital += exit_value
        
        # Record portfolio state
        self._record_portfolio_state(date)
        
        return True
    
    def get_performance_metrics(self):
        """Calculate overall performance metrics for the portfolio"""
        # Get current portfolio value
        current_portfolio = self.get_current_portfolio_value()
        
        # Calculate returns
        if not self.closed_trades and not self.open_trades:
            return {
                "total_trades": 0,
                "win_rate_pct": 0,
                "avg_return_pct": 0,
                "avg_win_pct": 0,
                "avg_loss_pct": 0,
                "profit_factor": 0,
                "avg_trade_duration": 0,
                "initial_capital": self.initial_capital,
                "final_value": current_portfolio,
                "total_return_pct": 0,
                "annualized_return_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown_pct": 0
            }
        
        # Calculate metrics from closed trades
        returns = [trade.calculate_return() for trade in self.closed_trades]
        durations = [(trade.exit_date - trade.entry_date).days for trade in self.closed_trades]
        
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        win_rate = len(wins) / len(returns) if returns else 0
        avg_return = sum(returns) / len(returns) if returns else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate portfolio metrics
        portfolio_df = self.get_portfolio_value_df()
        
        # Calculate annualized return
        if len(portfolio_df) > 1:
            start_date = portfolio_df.index[0]
            end_date = portfolio_df.index[-1]
            years = (end_date - start_date).days / 365.25
            total_return = (current_portfolio / self.initial_capital) - 1
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            annualized_return = 0
        
        # Calculate drawdowns and Sharpe ratio
        if len(portfolio_df) > 1:
            portfolio_df['Daily_Return'] = portfolio_df['Value'].pct_change()
            sharpe_ratio = portfolio_df['Daily_Return'].mean() / portfolio_df['Daily_Return'].std() * (252 ** 0.5) if portfolio_df['Daily_Return'].std() > 0 else 0
            
            portfolio_df['Peak'] = portfolio_df['Value'].cummax()
            portfolio_df['Drawdown'] = (portfolio_df['Value'] / portfolio_df['Peak']) - 1
            max_drawdown = abs(portfolio_df['Drawdown'].min()) * 100 if not portfolio_df['Drawdown'].empty else 0
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Return metrics
        return {
            "total_trades": len(self.closed_trades),
            "win_rate_pct": win_rate * 100,  # As percentage
            "avg_return_pct": avg_return * 100,  # As percentage
            "avg_win_pct": avg_win * 100,  # As percentage
            "avg_loss_pct": avg_loss * 100,  # As percentage
            "profit_factor": profit_factor,
            "avg_trade_duration": avg_duration,
            "initial_capital": self.initial_capital,
            "final_value": current_portfolio,
            "total_return_pct": ((current_portfolio / self.initial_capital) - 1) * 100,  # As percentage
            "annualized_return_pct": annualized_return * 100,  # As percentage
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown
        }
        
    def update_trades(self, date, price_dict):
        """
        Update all open trades with new price data
        
        Parameters:
        -----------
        date : datetime
            Current date
        price_dict : dict
            Dictionary mapping ticker -> current price
        """
        # Skip if no price updates
        if not price_dict:
            return
            
        # Update each open trade
        closed_tickers = []
        for ticker, trade in self.open_trades.items():
            if ticker in price_dict:
                current_price = price_dict[ticker]
                exit_info = trade.update(date, current_price)
                
                if exit_info is not None:  # Trade was exited
                    effective_price = current_price * (1 - (self.slippage_pct * (1 if trade.position_type == 'LONG' else -1)))
                    exit_value = trade.shares * effective_price
                    # Remove commission calculation
                    net_exit_value = exit_value
                    
                    # Record the exit without commission costs
                    exit_record = {
                        'date': date,
                        'ticker': ticker,
                        'action': 'CLOSE',
                        'position_type': trade.position_type,
                        'price': effective_price,
                        'exit_value': net_exit_value,
                        'return_pct': trade.calculate_return() * 100,
                        'exit_reason': trade.exit_reason
                    }
                    self.trade_history.append(exit_record)
                    
                    # Move to closed trades and return capital
                    self.closed_trades.append(trade)
                    closed_tickers.append(ticker)
                    self.available_capital += net_exit_value
        
        # Remove closed trades from open trades
        for ticker in closed_tickers:
            del self.open_trades[ticker]
            
        # Update total portfolio value
        total_value = self.available_capital
        for trade in self.open_trades.values():
            total_value += trade._calculate_position_value()
        
        self.daily_portfolio_value[date] = total_value
        self.last_update_date = date
        
    def get_portfolio_value_df(self):
        """Get portfolio value as DataFrame"""
        if not self.daily_portfolio_value:
            return pd.DataFrame()
            
        df = pd.DataFrame(
            list(self.daily_portfolio_value.values()), 
            index=list(self.daily_portfolio_value.keys()),
            columns=['Value']
        )
        df.index.name = 'Date'
        return df
        
    def get_trade_history_df(self):
        """Get trade history as DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)
        
    def get_closed_trades_df(self):
        """Get closed trades summary as DataFrame"""
        if not self.closed_trades:
            return pd.DataFrame()
            
        trade_data = [trade.get_trade_summary() for trade in self.closed_trades]
        return pd.DataFrame(trade_data)
        
    def plot_portfolio_performance(self, benchmark_df=None, save_path=None):
        """
        Plot portfolio performance
        
        Parameters:
        -----------
        benchmark_df : pandas.DataFrame or None
            Optional benchmark data with 'Value' column
        save_path : str or None
            Path to save the figure
        """
        if not self.daily_portfolio_value:
            print("No portfolio data to plot")
            return
            
        portfolio_df = self.get_portfolio_value_df()
        
        # Calculate daily returns
        portfolio_df['Daily Return'] = portfolio_df['Value'].pct_change()
        
        # Calculate drawdowns
        portfolio_df['Peak'] = portfolio_df['Value'].cummax()
        portfolio_df['Drawdown'] = (portfolio_df['Value'] / portfolio_df['Peak']) - 1
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot portfolio value
        ax1 = axes[0]
        portfolio_df['Value'].plot(ax=ax1, label='Portfolio Value', color='blue')
        
        if benchmark_df is not None:
            # Normalize benchmark to same starting value
            bench_start = benchmark_df['Value'].iloc[0]
            port_start = portfolio_df['Value'].iloc[0]
            scale = port_start / bench_start
            benchmark_df['Scaled'] = benchmark_df['Value'] * scale
            benchmark_df['Scaled'].plot(ax=ax1, label='Benchmark', color='gray', linestyle='--')
            
        ax1.set_title('Portfolio Performance')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_xlabel('')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot daily returns
        ax2 = axes[1]
        portfolio_df['Daily Return'].plot(ax=ax2, kind='bar', color='green', alpha=0.5)
        ax2.set_title('Daily Returns')
        ax2.set_ylabel('Daily Return %')
        ax2.set_xlabel('')
        ax2.grid(True, alpha=0.3)
        
        # Plot drawdowns
        ax3 = axes[2]
        ax3.fill_between(portfolio_df.index, 0, portfolio_df['Drawdown'], color='red', alpha=0.3)
        ax3.set_title('Portfolio Drawdown')
        ax3.set_ylabel('Drawdown %')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def plot_trade_analysis(self, save_path=None):
        """
        Plot trade analysis charts
        
        Parameters:
        -----------
        save_path : str or None
            Path to save the figure
        """
        if not self.closed_trades:
            print("No closed trades to analyze")
            return
            
        trades_df = self.get_closed_trades_df()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot trade returns distribution
        ax1 = axes[0, 0]
        trades_df['return_pct'].plot(ax=ax1, kind='hist', bins=20, alpha=0.5, color='blue')
        ax1.axvline(x=0, color='red', linestyle='--')
        ax1.set_title('Trade Returns Distribution')
        ax1.set_xlabel('Return %')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative returns
        ax2 = axes[0, 1]
        trades_df.sort_values('exit_date', inplace=True)
        trades_df['Cumulative Return'] = (1 + trades_df['return_pct']/100).cumprod() - 1
        trades_df['Cumulative Return'].plot(ax=ax2)
        ax2.set_title('Cumulative Trade Returns')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative Return %')
        ax2.grid(True, alpha=0.3)
        
        # Plot win/loss by exit reason
        ax3 = axes[1, 0]
        reason_counts = trades_df.groupby(['exit_reason', trades_df['return_pct'] > 0]).size().unstack()
        if not reason_counts.empty:
            reason_counts.columns = ['Loss', 'Win']
            reason_counts.plot(ax=ax3, kind='bar', stacked=True)
            ax3.set_title('Win/Loss by Exit Reason')
            ax3.set_xlabel('Exit Reason')
            ax3.set_ylabel('Count')
            ax3.grid(True, alpha=0.3)
        
        # Plot returns by duration
        ax4 = axes[1, 1]
        ax4.scatter(trades_df['duration_days'], trades_df['return_pct'], alpha=0.7)
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_title('Trade Returns vs Duration')
        ax4.set_xlabel('Duration (days)')
        ax4.set_ylabel('Return %')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def _record_portfolio_state(self, date):
        """Record the current portfolio state for the given date"""
        total_value = self.available_capital
        for trade in self.open_trades.values():
            total_value += trade._calculate_position_value()
        
        self.daily_portfolio_value[date] = total_value
        self.last_update_date = date

    def get_current_portfolio_value(self):
        """Get the current total portfolio value"""
        if not self.daily_portfolio_value:
            return self.initial_capital
        
        return list(self.daily_portfolio_value.values())[-1]

    def check_position_balance(self):
        """
        Check if we have a balance of long and short positions
        """
        if not self.closed_trades:
            return {}
            
        position_types = [t.position_type for t in self.closed_trades]
        long_count = position_types.count('LONG')
        short_count = position_types.count('SHORT')
        
        balance = {
            'LONG': long_count,
            'SHORT': short_count,
            'long_pct': long_count/(long_count+short_count)*100 if long_count+short_count > 0 else 0,
            'short_pct': short_count/(long_count+short_count)*100 if long_count+short_count > 0 else 0
        }
        
        print(f"\nPosition balance check: LONG: {long_count} ({balance['long_pct']:.1f}%), "
              f"SHORT: {short_count} ({balance['short_pct']:.1f}%)")
        
        if long_count == 0 or short_count == 0:
            print("WARNING: Trading is one-sided! Consider rebalancing your strategy.")
        
        return balance

# Utility functions for exit strategies
def calculate_trailing_stop(price_series, window=14, multiplier=2.0):
    """
    Calculate a trailing stop based on Average True Range (ATR)
    
    Parameters:
    -----------
    price_series : pandas.Series
        Series of price data
    window : int
        Window for ATR calculation
    multiplier : float
        ATR multiplier
        
    Returns:
    --------
    pandas.Series
        Series of trailing stop values
    """
    # Calculate True Range
    high = price_series.rolling(2).max()
    low = price_series.rolling(2).min()
    tr = high - low
    
    # Average True Range
    atr = tr.rolling(window).mean()
    
    # Trailing stop (for long position)
    stop = price_series - (atr * multiplier)
    
    return stop

def volatility_adjusted_stop_loss(price, volatility, multiplier=2.0):
    """
    Calculate volatility-adjusted stop loss
    
    Parameters:
    -----------
    price : float
        Current price
    volatility : float
        Volatility measure (e.g., standard deviation)
    multiplier : float
        Volatility multiplier
        
    Returns:
    --------
    float
        Stop loss price
    """
    return price * (1 - multiplier * volatility)

def chandelier_exit(price_series, periods=22, atr_periods=14, multiplier=3.0):
    """
    Calculate Chandelier Exit - a trailing stop based on ATR from recent high
    
    Parameters:
    -----------
    price_series : pandas.Series
        Series of price data
    periods : int
        Lookback period for highest high
    atr_periods : int
        Period for ATR calculation
    multiplier : float
        ATR multiplier
        
    Returns:
    --------
    pandas.Series
        Series of chandelier exit values
    """
    # Calculate ATR
    high = price_series.rolling(2).max()
    low = price_series.rolling(2).min()
    tr = high - low
    atr = tr.rolling(atr_periods).mean()
    
    # Calculate highest high over lookback period
    highest_high = price_series.rolling(periods).max()
    
    # Chandelier exit (for long position)
    chandelier = highest_high - (atr * multiplier)
    
    return chandelier