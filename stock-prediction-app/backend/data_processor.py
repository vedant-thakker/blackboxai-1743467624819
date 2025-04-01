import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(stock_symbol):
    """
    Load and preprocess stock data from CSV file
    Returns a pandas DataFrame with processed data
    """
    try:
        # In a real implementation, this would load from a database or API
        # For demo purposes, we'll generate synthetic data
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=3650, freq='D')
        base_price = np.random.randint(500, 1500)
        daily_returns = np.random.normal(0.001, 0.02, 3650)
        price_series = base_price * (1 + daily_returns).cumprod()
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': price_series
        })
        
        # Convert date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['Close'] = scaler.fit_transform(df[['Close']])
        
        return df, scaler
        
    except Exception as e:
        raise Exception(f"Data processing error: {str(e)}")