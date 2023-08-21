"""
This file contains the functions for loading, saving and pre-processing the MNIST dataset, along with some functions fol plotting the digits of the dataset etc..
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm

# LOADING
def load_dataset():
    X = pd.read_parquet('dataset/X.parquet')
    y = pd.read_parquet('dataset/y.parquet').squeeze() 

    return X,y
    
def load_PCA_datasets(max_pca_dim:int):
    """
    This function loads the reduced datasets from the "dataset" folder.

    INPUT:
    - max_pca_dim = int, i.e. the maximum PCA dimension.

    OUTPUT: ( {pca_dim : pca_dataset} , y )
    """
    
    y = pd.read_parquet('dataset/validation/y.parquet').squeeze() 
    datasets = {}
    
    for i in tqdm(range(2,max_pca_dim+10,10), desc="Loading the PCA datasets.."):
        datasets[i] = pd.read_parquet("dataset/PCA_"+str(i)+".parquet")
        
    return datasets, y

# PRE-PROCESSING
def train_valid_test_split(X:pd.DataFrame, y:pd.Series, test_size:float, validation_size:float):
    """
    This function performs a train-validation-test split of the dataset.
    
    INPUT:
    - X, i.e. the feature vector;
    - y, i.e. the label vector;
    - test_size, i.e. the size of the test set;
    - validation_size, i.e. the size of the validation set.

    OUTPUT:
    - X_train, i.e. the feature vector for the training phase;
    - y_train, i.e. the label vector for the training phase;
    - X_valid, i.e. the feature vector for the validation phase;
    - y_valid, i.e. the label vector for the validation phase;
    - X_test, i.e. the feature vector for the testing phase;
    - y_test, i.e. the label vector for the testing phase.
    """

    X_train_80, X_test, y_train_80, y_test = train_test_split(X, y, test_size = test_size, random_state = 1)
    X_train, X_valid , y_train, y_valid = train_test_split(X_train_80, y_train_80, test_size = validation_size, random_state = 1)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def apply_PCA(X:pd.DataFrame, y:pd.Series, max_pca_dim:int, dataset_percentage:float, test_size:float, validation_size:float):
    """
    This function applies some PCA transformations to a fraction of the MNIST dataset.\nFor the aim of the project, the PCA dimension varies from 2 to 200.
    """
    
    # random sampling
    r = np.random.RandomState(1)
    indexes = r.choice(70000, int(70000*dataset_percentage),replace=False)
    
    # saving the sampled dataset 
    X.iloc[indexes].to_parquet("dataset/X.parquet")
    y[indexes].to_frame().to_parquet("dataset/y.parquet")

    for i in tqdm(range(2,max_pca_dim+10,10), desc="Applying PCA transformation.."):
        pca = PCA(n_components=i)
        
        # applying the PCA transformation to the sampled dataset
        df = pd.DataFrame(pca.fit_transform(X),columns=["pca_"+str(x) for x in range(1,i+1)])
        df = df.iloc[indexes]

        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df,y[indexes].to_frame(),test_size,validation_size)
        
        # saving the label vectors
        if not Path("dataset/train/y.parquet").is_file():
            y_train.to_parquet("dataset/train/y.parquet")

        if not Path("dataset/valid/y.parquet").is_file():
            y_valid.to_parquet("dataset/valid/y.parquet")

        if not Path("dataset/test/y.parquet").is_file():
            y_test.to_parquet("dataset/test/y.parquet")

        # saving the transformed dataset 
        X_train.to_parquet("dataset/train/X_"+str(i)+".parquet")
        X_valid.to_parquet("dataset/validation/X_"+str(i)+".parquet")
        X_test.to_parquet("dataset/test/X_"+str(i)+".parquet")

def new_apply_PCA(X:pd.DataFrame, y:pd.Series, max_pca_dim:int, dataset_percentage:float, test_size:float, validation_size:float):
    """
    This function applies some PCA transformations to a fraction of the MNIST dataset.\nFor the aim of the project, the PCA dimension varies from 2 to 200.
    """
    
    # random sampling
    r = np.random.RandomState(1)
    indexes = r.choice(70000, int(70000*dataset_percentage),replace=False)
    
    # saving the sampled dataset 
    X.iloc[indexes].to_parquet("dataset/X.parquet")
    y[indexes].to_frame().to_parquet("dataset/y.parquet")

    # splitting into train, validation and test
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(X.iloc[indexes], y[indexes].to_frame(), test_size, validation_size)
    y_train.to_parquet("dataset/train/y.parquet")
    y_valid.to_parquet("dataset/valid/y.parquet")
    y_test.to_parquet("dataset/test/y.parquet")

    for i in tqdm(range(2,max_pca_dim+10,10), desc="Applying PCA transformation.."):
        pca = PCA(n_components=i)
        
        # applying the PCA transformation to the train, validation and test datasets
        df_train = pd.DataFrame(pca.fit_transform(X_train),columns=["pca_"+str(x) for x in range(1,i+1)])
        df = df_train.iloc[indexes]

        X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df,y[indexes].to_frame(),test_size,validation_size)
        
        # saving the transformed dataset 
        X_train.to_parquet("dataset/train/X_"+str(i)+".parquet")
        y_train.to_parquet("dataset/train/y_"+str(i)+".parquet")
        X_valid.to_parquet("dataset/validation/X_"+str(i)+".parquet")
        y_valid.to_parquet("dataset/validation/y_"+str(i)+".parquet")
        X_test.to_parquet("dataset/test/X_"+str(i)+".parquet")
        y_test.to_parquet("dataset/test/y_"+str(i)+".parquet")

# PLOTS
def plot_digits(iter, X, y):
    fig, axs = plt.subplots(10, iter=15)
    
    for digit in range(10):
        for x in range(iter):
            digit_index = y[y == digit].index[x]
            digit_pixels = np.array(X.iloc[digit_index]).reshape(28, 28)
            axs[digit,x].imshow(digit_pixels)
            axs[digit,x].axis('off')