from flask import Flask
from flask import request
from flask import jsonify
import joblib

import pickle

def load(filename):
        return joblib.load(filename)


dv = load('dv.bin')
model = load('model1.bin')

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    cc_data = request.get_json()

    X = dv.transform([cc_data])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)