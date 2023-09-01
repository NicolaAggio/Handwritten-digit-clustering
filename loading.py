"""
This file contains the functions for loading the MNISt dataset and the main files used in the project.
"""
import pandas as pd
import pickle

from typing import List
from sklearn.datasets import fetch_openml
from pathlib import Path
from tqdm import tqdm

# LOADING
def load_MNIST_dataset():
    """
    This function loads the MNIST dataset from the "dataset" folder.
    """
    X = pd.read_parquet('dataset/X.parquet')
    y = pd.read_parquet('dataset/y.parquet').squeeze() 

    return X,y

def load_reduced_dataset(dataset_percentage:float):
    """
    This function loads the reduced dataset from the corresponding folder.

    INPUT: dataset_percentage, i.e. the percentage of dataset that is considered.

    OUTPUT:
    - X, i.e. the feature vector;
    - y, i.e. the label vector.
    """
    X = pd.read_parquet('dataset/' + str(dataset_percentage) + '/X.parquet')
    y = pd.read_parquet('dataset/' + str(dataset_percentage) + '/y.parquet').squeeze() 

    return X,y

def download_dataset():
    """
    This function downloads the MNIST dataset, if not present in the "dataset" folder, otherwise it loads it from the same folder.

    OUTPUT:
    - X, i.e. the feature vector;
    - y, i.e. the label vector.
    """
    if not Path("dataset/X.parquet").is_file() and not Path("dataset/y.parquet").is_file():
        X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = y.astype(int)
        X = X/255

        X.to_parquet("dataset/X.parquet")
        y.to_frame().to_parquet("dataset/y.parquet")
    else:
        X,y = load_MNIST_dataset()
    
    return X,y

def load_PCA_2(dataset_percentage:float):
    """
    This function loads the test dataset corresponding to PCA dimension 2 from the corresponding folder.

    OUTPUT:
    - X_test, i.e. the testing feature vector;
    - y_test, i.e. the testing label vector.
    """
    y_test = pd.read_parquet('dataset/' + str(dataset_percentage) + '/test/y.parquet').squeeze() 
    X_test = pd.read_parquet("dataset/" + str(dataset_percentage) + "/test/X_2.parquet")

    return X_test, y_test

def load_PCA_train_sets(pca_dimensions:List[int], dataset_percentage:float):
    """
    This function loads the transformed training datasets from the "dataset/train" folders.

    INPUT:
    - pca_dimensions, i.e. the PCA values;
    - dataset_percentage, i.e. the percentage of dataset that is considered.

    OUTPUT: (X_train, y_train), where:
    - X_train, i.e. {PCA_dim : X_train, for each PCA_dim};
    - y_train, i.e. the training feature vector;
    """
    
    y_train = pd.read_parquet('dataset/' + str(dataset_percentage) + '/train/y.parquet').squeeze() 
    X_train = {}
    
    for i in tqdm(pca_dimensions, desc="Loading the PCA training datasets.."):
        X_train[i] = pd.read_parquet("dataset/" + str(dataset_percentage) + "/train/X_" + str(i) + ".parquet")
        
    return X_train, y_train

def load_PCA_test_sets(pca_dimensions:List[int], dataset_percentage:float):
    """
    This function loads the transformed testing datasets from the "dataset/test" folders.

    INPUT:
    - pca_dimensions = int, i.e. the PCA values;
    - dataset_percentage, i.e. the percentage of dataset that is considered.

    OUTPUT: (X_test, y_test), where:
    - X_test, i.e. {PCA_dim : X_test, for each PCA_dim};
    - y_test, i.e. the testing feature vector;
    """
    
    y_test = pd.read_parquet('dataset/' + str(dataset_percentage) + '/test/y.parquet').squeeze() 
    X_test = {}
    
    for i in tqdm(pca_dimensions, desc="Loading the PCA testing datasets.."):
        X_test[i] = pd.read_parquet("dataset/" + str(dataset_percentage) + "/test/X_" + str(i) + ".parquet")
        
    return X_test, y_test

def load_tuning_results(model_name:str):
    """
    This function loads the results of the tuning phase of the giiven model from the "GridSearch_tuning/" folder.

    INPUT: model_name, i.e. the model;

    OUTPUT: tuning results of the specified model.
    """
    with open("GridSearch_tuning/" + model_name + ".pkl","rb") as file:
        return pickle.load(file)
