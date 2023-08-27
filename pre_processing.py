"""
This file contains the functions for saving and pre-processing the MNIST dataset.
"""
import pandas as pd
import numpy as np
import pickle

from typing import List
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm

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

            # saving the PCA object
            with open("PCA/"+str(i)+".pkl", 'wb') as out:
                pickle.dump(pca, out, pickle.HIGHEST_PROTOCOL)  