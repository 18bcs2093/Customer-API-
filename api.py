import pandas as pd
from flask import Flask, request, jsonify
from model import CustomerReturnPredictionModel
app = Flask(__name__)

model = CustomerReturnPredictionModel(pd.read_csv('customer_data.csv'))

@app.route('/predict', methods=['POST'])
def predict():
    customer_id = request.json['customer_id']
    purchase_history = request.json['purchase_history']

    predictions = model.predict(customer_id, purchase_history)

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
