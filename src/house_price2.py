# -*- coding: utf-8 -*-
"""house_price2.py

Model training script for predicting house prices using a linear regression model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the data
data = pd.read_csv("house_prices.csv")  # Make sure this file exists in your working directory
print(data.head(10))

# Check for missing values
missing = data.isnull().sum()
print("Missing values per column:")
print(missing)

# Prepare the data
y = data.iloc[:, 2:3]
x = data.iloc[:, 3:19]
x = pd.get_dummies(x)  # Convert categorical variables to dummy/indicator variables

# Save feature names for later use
feature_names = list(x.columns)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)
print("Predictions on the test set:")
print(y_pred)

# Print model scores
print(f"Train Score: {model.score(x_train, y_train)}")
print(f"Test Score: {model.score(x_test, y_test)}")

# Save the model
with open('house_price2.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


