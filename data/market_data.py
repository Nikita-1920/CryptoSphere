import yfinance as yf
import pandas as pd
import numpy as np
import functools

@functools.lru_cache(maxsize=1)
def get_exchange_rate(currency_pair='USDINR=X'):
    """Fetch live exchange rate."""
    try:
        ticker = yf.Ticker(currency_pair)
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
    return 83.0 # Fallback INR rate

def fetch_crypto_data(symbol, start_date=None, end_date=None, interval='1d', period='1y', currency='USD'):
    """Fetch historical data for a given cryptocurrency symbol and apply feature engineering."""
    try:
        ticker = yf.Ticker(symbol)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)
            
        if df.empty:
            return None
            
        # Currency Conversion
        if currency == 'INR':
            rate = get_exchange_rate('USDINR=X')
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col] * rate
        
        # --- Feature Engineering ---
        # Returns
        df['Return'] = df['Close'].pct_change()
        
        # Moving Averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_14'] = df['Close'].rolling(window=14).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility
        df['Volatility'] = df['Return'].rolling(window=14).std()
        
        # Momentum (Close vs Close 14 days ago)
        df['Momentum'] = df['Close'] - df['Close'].shift(14)
        
        # Lag Features
        df['Lag_1'] = df['Close'].shift(1)
        df['Lag_2'] = df['Close'].shift(2)
        
        # Rolling Statistics
        df['Rolling_Mean_14'] = df['Close'].rolling(window=14).mean()
        df['Rolling_Std_14'] = df['Close'].rolling(window=14).std()
        
        # Trend (1 if closing higher than open, else 0)
        df['Trend'] = (df['Close'] > df['Open']).astype(int)
        
        # Drop initial NaN rows due to rolling windows
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
