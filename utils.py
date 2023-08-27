"""
This file contains the functions for loading, saving and pre-processing the MNIST dataset, along with some functions fol plotting the digits of the dataset etc..
"""
import pandas as pd
import numpy as np
import pickle

from typing import List
from sklearn.datasets import fetch_openml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
    """
    X = pd.read_parquet('dataset/' + str(dataset_percentage) + '/X.parquet')
    y = pd.read_parquet('dataset/' + str(dataset_percentage) + '/y.parquet').squeeze() 

    return X,y

def download_dataset():
    """
    This function downloads the MNIST dataset, if not present in the "dataset" folder, otherwise it loads it from the same folder.
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
    with open("GridSearch_tuning/" + model_name + ".pkl","rb") as file:
        return pickle.load(file)

# PRE-PROCESSING

def split(X:pd.DataFrame, y:pd.Series, test_size:float):
    """
    This function performs a train-test splitting of the dataset.

    INPUT:
    - X, i.e. the feature dataframe;
    - y, i.e. the label vector;
    - test_size, i.e. the size of the test set.

    OUTPUT:
    - X_train, i.e. the training feature dataframe;
    - y_train, i.e. the training label vector;
    - X_test, i.e. the testing feature dataframe;
    - y_test, i.e. the testing label vector;
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)
    return X_train, y_train, X_test, y_test

def apply_PCA(X:pd.DataFrame, y:pd.Series, pca_dimensions:List[int], dataset_percentage:float, test_size:float):
    """
    This function applies some PCA transformations to a fraction of the MNIST dataset.\nFor the aim of the project, the dataset percentage is set to 50% and the PCA dimension varies from 2 to 200.
    
    INPUT:
    - X, i.e. the feature dataframe;
    - y, i.e. the label vector;
    - max_pca_dim, i.e. the maximum dimension of PCA;
    - dataset_percentage, i.e. the percentage of dataset that is used;
    - test_size, i.e. the size of the test set.
    """

    # random sampling
    r = np.random.RandomState(1)
    indexes = r.choice(70000, int(70000*dataset_percentage),replace=False)
    
    # saving the sampled dataset 
    X.iloc[indexes].to_parquet("dataset/" + str(dataset_percentage) + "/X.parquet")
    y[indexes].to_frame().to_parquet("dataset/" + str(dataset_percentage) + "/y.parquet")

    for i in tqdm(pca_dimensions, desc = "Applying PCA transformation.."):
        if not Path("dataset/" + str(dataset_percentage) + "/train/X_" + str(i) + ".parquet").is_file() or not Path("dataset/" + str(dataset_percentage) + "/test/X_" + str(i) + ".parquet").is_file():
            pca = PCA(n_components=i)
            
            # applying the PCA transformation to the sampled dataset
            df = pd.DataFrame(pca.fit_transform(X),columns=["pca_"+str(x) for x in range(1,i+1)])
            df = df.iloc[indexes]

            X_train, y_train, X_test, y_test = split(df,y[indexes].to_frame(),test_size)
            
            # saving the label vectors
            y_train.to_parquet("dataset/" + str(dataset_percentage) + "/train/y.parquet")
            y_test.to_parquet("dataset/" + str(dataset_percentage) + "/test/y.parquet")

            # saving the transformed dataset 
            X_train.to_parquet("dataset/" + str(dataset_percentage) + "/train/X_"+str(i)+".parquet")
            X_test.to_parquet("dataset/" + str(dataset_percentage) + "/test/X_"+str(i)+".parquet") 

            with open("PCA/"+str(i)+".pkl", 'wb') as out:
                pickle.dump(pca, out, pickle.HIGHEST_PROTOCOL)           