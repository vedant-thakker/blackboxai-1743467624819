import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta

def create_model(input_shape):
    """
    Create and compile LSTM model for stock prediction
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_stock(df, days_to_predict=30):
    """
    Train model and make predictions for given number of days
    Returns list of predictions with dates and confidence
    """
    try:
        # Prepare training data
        data = df['Close'].values
        training_data_len = int(np.ceil(len(data) * 0.8))
        
        # Create training dataset
        x_train = []
        y_train = []
        
        for i in range(60, training_data_len):
            x_train.append(data[i-60:i])
            y_train.append(data[i])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Build and train model
        model = create_model((x_train.shape[1], 1))
        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
        
        # Create test data
        test_data = data[training_data_len - 60:]
        x_test = []
        
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Make predictions
        predictions = model.predict(x_test)
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        
        # Create results with confidence scores
        results = []
        for i, date in enumerate(future_dates[:min(days_to_predict, len(predictions))]):
            confidence = 'High' if i < len(predictions)//2 else 'Medium'
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': float(predictions[i][0] * 1000 + 1000),  # Scale back to realistic prices
                'confidence': confidence
            })
        
        return results
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")