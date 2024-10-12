import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import sys
sys.path.append('./src/data/')
from check_structure import check_existing_file


def main():
    """ Runs data processing scripts to analyze data 
    from (../processed_data and ../ scaled_data) 
    """
    logger = logging.getLogger(__name__)
    logger.info('Looking for best parameters  from scaled data')

    gridsearching()

def gridsearching():

    # Setting folders and files
    input_filepath_X_train = "./data/scaled_data/X_train_scaled.csv"
    input_filepath_y_train = "./data/processed_data/y_train.csv"
    output_folderpath = "./models/"

    # Import datasets
    print("Importing datasets...")
    X_train = import_dataset(input_filepath_X_train, sep=",")
    y_train = import_dataset(input_filepath_y_train, sep=",")
    y_train = np.ravel(y_train)

    # Normalize the X data
    print("Instantiate a random forest regressor...")
    rfr = RandomForestRegressor(random_state=0, n_jobs=-1)
    #parameters = {'n_estimators':[50], 'max_depth':[2]}
    parameters = {
        'n_estimators':[50,100,200], 
        'max_depth':[None, 5, 10], 
        'min_samples_split': [2, 5, 10], 
        'min_samples_leaf': [2, 5, 10], 
        'max_features': ['sqrt', 'log2', None]
    }
    clf = GridSearchCV(rfr, parameters, cv=5)
    print("Running the Grid Search CV...")
    clf.fit(X_train, y_train)
    best_params = clf.best_params_
    print(f"Best parameters for random forest regressor: {best_params}")

    # Save the best parameters to a pickle file
    print("Saving the best params found by the Grid Search CV...")
    save_params(best_params, output_folderpath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def save_params(best_params, output_folderpath):
    # Save dataframes to their respective output file paths
    output_filepath = os.path.join(output_folderpath, 'best_params.pkl')
    if check_existing_file(output_filepath):
        joblib.dump(best_params, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
