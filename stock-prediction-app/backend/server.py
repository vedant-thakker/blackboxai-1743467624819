from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from model import predict_stock
from data_processor import load_and_preprocess_data
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)  # Enable CORS for all routes

# Serve frontend files
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/results')
def results_page():
    return render_template('results.html')

# API endpoints
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        stock_symbol = data.get('symbol', 'RELIANCE.NS')
        timeframe = int(data.get('timeframe', 30))
        
        # Load and preprocess data
        df, scaler = load_and_preprocess_data(stock_symbol)
        
        # Get predictions
        predictions = predict_stock(df, timeframe)
        
        # Format response
        response = {
            'status': 'success',
            'symbol': stock_symbol,
            'predictions': predictions,
            'historical': [{
                'date': date.strftime('%Y-%m-%d'),
                'price': float(price * 1000 + 1000)  # Scale back to realistic prices
            } for date, price in df['Close'].items()][-100:]  # Last 100 days
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)