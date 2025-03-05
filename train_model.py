import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set file path
csv_filename = "car_price.csv"

# Check if CSV file exists
if not os.path.exists(csv_filename):
    print(f"Error: {csv_filename} not found!")
    exit()

# Load dataset
data = pd.read_csv(csv_filename)
print("File loaded successfully!")

# Convert column names to lowercase to avoid case-sensitive issues
data.columns = data.columns.str.lower()
print("Columns in CSV:", list(data.columns))  # Debugging step

# Select features and target variable
X = data[['brand', 'model', 'year', 'engine_size', 'fuel_type', 'transmission', 
          'mileage', 'doors', 'owner_count']]  
y = data['price']  # Now it works

# Fill missing categorical values before encoding
for col in ['brand', 'model', 'fuel_type', 'transmission']:
    X[col] = X[col].fillna("Unknown")  # Fixed SettingWithCopyWarning

# Encode categorical features
categorical_features = ['brand', 'model', 'fuel_type', 'transmission']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)  # Transform the data

# Convert the NumPy array to a DataFrame
X = pd.DataFrame(X)

# Fill missing numerical values
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save model + preprocessor
with open("linear_model.pkl", "wb") as file:
    pickle.dump((model, preprocessor), file)

print("Model saved as 'linear_model.pkl'")
