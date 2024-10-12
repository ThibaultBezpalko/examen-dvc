
import sklearn
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

print(f"Jpblib version: {joblib.__version__}")

# Loading datasets
print("Loading training datasets")
X_train = pd.read_csv('./data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv')
y_train = np.ravel(y_train)

# Retrieve best parameters from GridSearchCV
best_params = joblib.load('./models/best_params.pkl')

# Train the model
print("Training model")
rf_regressor = RandomForestRegressor(**best_params)
rf_regressor.fit(X_train, y_train)

# Save the trained model to a file
model_filename = './models/trained_model.joblib'
joblib.dump(rf_regressor, model_filename)
print("Model trained and saved successfully.")
