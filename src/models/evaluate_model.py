import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.append('./src/data/')
from check_structure import check_existing_file, check_existing_folder

# Loading datasets
print("Loading training datasets")
X_test = pd.read_csv('./data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('./data/processed_data/y_test.csv')
y_test = np.ravel(y_test)
X_train = pd.read_csv('./data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv')
y_train = np.ravel(y_train)

def main(repo_path):
    # Load the trained model
    print("Loading trained model...")
    model = load(repo_path / "./models/trained_model.pkl")

    # Predictions
    print("Predicting...")
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Create folder if necessary
    output_folderpath = "./data/predicted_data/"
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

    # Save predictions
    output_filepath = os.path.join(output_folderpath, 'predictions.csv')
    if check_existing_file(output_filepath):
        y_test_pred_df = pd.DataFrame(y_test_pred, columns=['silica_concentrate'])
        y_test_pred_df.to_csv(output_filepath, index=False)

    # Calcul des métriques
    print("Evaluating...")
    # jeu de test 
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    
    # jeu d'entraînement 
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)

    metrics = {
        "r2": {
            "test": r2_test,
            "train": r2_train
        },
        "mean absolute error": {
            "test": mae_test,
            "train": mae_train
        },
        "mean squared error": {
            "test": mse_test,
            "train": mse_train
        },
        "root mean squared error": {
            "test": rmse_test,
            "train": rmse_train
        }
        }
    accuracy_path = repo_path / "./metrics/scores.json"
    accuracy_path.write_text(json.dumps(metrics))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)