import numpy as np
import pandas as pd

def _get_portfolio_returns(portfolio_df, historical_data_dict):
    """
    Helper function to align historical data and calculate daily portfolio returns.
    """
    # Create a DataFrame of daily closing prices for all assets in the portfolio
    price_df_list = []
    
    for _, row in portfolio_df.iterrows():
        asset = row['Asset']
        amount = row['Amount']
        
        if asset in historical_data_dict and not historical_data_dict[asset].empty:
            df = historical_data_dict[asset]
            # Multiply historical price by current amount held to get historical value of this position
            s = df['Close'] * amount
            s.name = asset
            price_df_list.append(s)
            
    if not price_df_list:
        return None
        
    # Combine into a single dataframe and forward fill any missing dates
    portfolio_history = pd.concat(price_df_list, axis=1).fillna(method='ffill')
    
    # Calculate total portfolio value over time
    portfolio_total = portfolio_history.sum(axis=1)
    
    # Calculate daily returns
    daily_returns = portfolio_total.pct_change().dropna()
    return daily_returns

def calculate_portfolio_risk(portfolio_df, historical_data_dict, current_total_value, risk_free_rate=0.045):
    """
    Calculates Volatility, Sharpe Ratio, and 95% Value at Risk (VaR).
    """
    daily_returns = _get_portfolio_returns(portfolio_df, historical_data_dict)
    
    if daily_returns is None or len(daily_returns) < 10:
        return 0.0, 0.0, 0.0
        
    # Annualized Volatility (assuming 365 crypto trading days)
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(365)
    
    # Sharpe Ratio
    annual_return = daily_returns.mean() * 365
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # Historical VaR (95% confidence)
    # The 5th percentile of historical daily returns
    var_95_pct = np.percentile(daily_returns, 5)
    
    # Dollar VaR
    var_95_dollar = current_total_value * var_95_pct
    
    return annual_vol, sharpe_ratio, var_95_dollar

def run_monte_carlo(portfolio_df, historical_data_dict, current_total_value, days=30, simulations=1000):
    """
    Runs a simple Monte Carlo simulation (Geometric Brownian Motion) on the total portfolio value.
    Returns a DataFrame with the 5th, 50th, and 95th percentile paths.
    """
    daily_returns = _get_portfolio_returns(portfolio_df, historical_data_dict)
    
    if daily_returns is None or len(daily_returns) < 10 or current_total_value <= 0:
        return pd.DataFrame()
        
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    
    # Generate random shocks
    # Shape: (days, simulations)
    random_shocks = np.random.normal(mu, sigma, (days, simulations))
    
    # Convert shocks to price multipliers
    price_multipliers = 1 + random_shocks
    
    # Calculate simulated portfolio paths
    # Starting at current_total_value
    paths = np.zeros_like(price_multipliers)
    paths[0] = current_total_value * price_multipliers[0]
    
    for t in range(1, days):
        paths[t] = paths[t-1] * price_multipliers[t]
        
    # Calculate percentiles across all simulations at each day
    percentile_5 = np.percentile(paths, 5, axis=1)
    percentile_50 = np.percentile(paths, 50, axis=1)
    percentile_95 = np.percentile(paths, 95, axis=1)
    
    # Create output dataframe
    dates = pd.date_range(start=pd.Timestamp.today(), periods=days)
    
    mc_df = pd.DataFrame({
        'Date': dates,
        'P5 (Pessimistic)': percentile_5,
        'P50 (Expected)': percentile_50,
        'P95 (Optimistic)': percentile_95
    }).set_index('Date')
    
    return mc_df
