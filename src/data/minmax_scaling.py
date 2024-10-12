import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from sklearn.preprocessing import MinMaxScaler
#sys.path.append('???')
from check_structure import check_existing_file


def main():
    """ Runs data processing scripts to turn raw data from (../raw_data) into
        cleaned data ready to be analyzed (saved in../preprocessed_data).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final scaled data set from raw data')

    scale_data()

def scale_data():

    input_filepath_X_train = "./data/processed_data/X_train.csv"
    input_filepath_X_test = "./data/processed_data/X_test.csv"
    output_filepath = "./data/processed_data/"

    # Import datasets
    X_train = import_dataset(input_filepath_X_train, sep=",")
    X_test = import_dataset(input_filepath_X_test, sep=",")

    # Normalize the X data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)


    # Save dataframes to their respective output file paths
    save_dataframes(X_train_scaled_df, X_test_scaled_df, output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def save_dataframes(X_train_scaled_df, X_test_scaled_df, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled_df, X_test_scaled_df], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
