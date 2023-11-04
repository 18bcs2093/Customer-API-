import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class CustomerReturnPredictionModel:
    def __init__(self, data):
        self.data = data
        self.model = LogisticRegression()

    def train(self):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop('target', axis=1), self.data['target'], test_size=0.25)

        # Train the model on the training set
        self.model.fit(X_train, y_train)

    def predict(self, X):
        # Predict the likelihood of customer return and product purchase
        y_pred = self.model.predict_proba(X)[:, 1]

        return y_pred

    def recommend_products(self, customer_id):
        # Get the customer's purchase history
        customer_history = self.data[self.data['customer_id'] == customer_id]['product_id']

        # Get the most popular products purchased by customers with similar purchase history
        similar_customer_history = self.data[self.data['product_id'].isin(customer_history)]['customer_id'].unique()
        most_popular_products = similar_customer_history.value_counts().sort_values(ascending=False).index.tolist()

        return most_popular_products

