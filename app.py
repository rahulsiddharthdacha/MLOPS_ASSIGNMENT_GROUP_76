from flask import Flask, request, jsonify
from flask_pydantic import validate
from pydantic import BaseModel, confloat
from prometheus_flask_exporter import PrometheusMetrics
import joblib
import sqlite3
import json 
import logging
from datetime import datetime
import os 
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('model.joblib')


# Logging configuration
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SQL Database setup
DB_PATH = "logs/predictions.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
               CREATE TABLE IF NOT EXISTS predictions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp TEXT,
                   input_data TEXT,
                   prediction REAL
               )
               ''')
conn.commit
conn.close()

# flask + prometheus

app = Flask(__name__)
metrics = PrometheusMetrics(app, group_by='endpoint')
metrics.info('app_info', 'Housing Price Prediction App', version='1.0.0')



@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(np.array(data['input']).reshape(1,-1))
    return jsonify({'prediction':prediction.tolist()})

@app.route('/metric_summary', methods=['GET'])
def metric_summary():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM predictions')
    count = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(prediction) FROM predictions')
    avg_prediction = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_predictions': count,
        'average_prediction': avg_prediction
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    data = request.json
    df = pd.DataFrame(data['data'])
    
    if 'median_house_value' not in df.columns:
        return jsonify({'error': 'Target column "median_house_value" not found in the dataset.'}), 400
    
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']
    
    model.fit(X, y)
    joblib.dump(model, 'model.joblib')
    
    return jsonify({'message': 'Model retrained successfully.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0' )