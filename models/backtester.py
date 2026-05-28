import pandas as pd
import numpy as np

def run_backtest(df, strategy_name, initial_capital=10000.0, transaction_fee=0.001):
    """
    Runs a simple vectorized backtest on the given DataFrame.
    Assumes df has 'Close' column and is sorted by Date.
    
    Returns:
        equity_curve (pd.DataFrame): Date indexed dataframe with 'Equity', 'Drawdown'
        metrics (dict): Summary metrics
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return None, {}
        
    df = df.copy()
    
    # Ensure indicators exist if needed
    if strategy_name == "MACD Crossover":
        if 'MACD' not in df.columns or 'Signal_Line' not in df.columns:
            # Simple calculation if missing
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
    elif strategy_name == "RSI Mean Reversion":
        if 'RSI' not in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

    # Generate Signals
    df['Position'] = 0 # 1 for Long, 0 for Flat
    
    if strategy_name == "Buy & Hold":
        df['Position'] = 1
        
    elif strategy_name == "MACD Crossover":
        # Buy when MACD > Signal, Sell when MACD < Signal
        df['Position'] = np.where(df['MACD'] > df['Signal_Line'], 1, 0)
        
    elif strategy_name == "RSI Mean Reversion":
        # Buy when RSI < 30, Hold until RSI > 70
        # This requires iterative logic to hold state, but we can approximate:
        # 1 when RSI < 30, 0 when RSI > 70, forward fill in between
        conditions = [
            df['RSI'] < 30,
            df['RSI'] > 70
        ]
        choices = [1, 0]
        df['Signal'] = np.select(conditions, choices, default=np.nan)
        df['Position'] = df['Signal'].ffill().fillna(0)

    # Calculate returns
    df['Market_Return'] = df['Close'].pct_change()
    
    # We apply the position from yesterday to today's return to avoid look-ahead bias
    df['Strategy_Return'] = df['Position'].shift(1) * df['Market_Return']
    
    # Calculate transaction costs when position changes
    df['Trades'] = df['Position'].diff().abs()
    df['Cost'] = df['Trades'] * transaction_fee
    
    # Net return
    df['Net_Strategy_Return'] = df['Strategy_Return'] - df['Cost']
    
    # Calculate Equity Curve
    df['Equity'] = initial_capital * (1 + df['Net_Strategy_Return'].fillna(0)).cumprod()
    
    # Calculate Drawdown
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
    
    # Metrics
    total_return = (df['Equity'].iloc[-1] - initial_capital) / initial_capital
    max_drawdown = df['Drawdown'].min()
    
    # Win rate calculation (profitable trades)
    # A trade is a continuous period of Position=1. We can approximate win rate by looking at daily returns while long.
    # Better: win rate of days
    winning_days = len(df[df['Net_Strategy_Return'] > 0])
    losing_days = len(df[df['Net_Strategy_Return'] < 0])
    total_active_days = winning_days + losing_days
    win_rate = winning_days / total_active_days if total_active_days > 0 else 0
    
    total_trades = df['Trades'].sum() / 2 # Buy and Sell makes 1 round trip trade
    
    metrics = {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Win Rate (Days)": win_rate,
        "Total Trades": int(total_trades)
    }
    
    equity_curve = df[['Equity', 'Drawdown', 'Position', 'Close']].copy()
    
    return equity_curve, metrics
