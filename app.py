from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.joblib')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(np.array(data['input']).reshape(1,-1))
    return jsonify({'prediction':prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')