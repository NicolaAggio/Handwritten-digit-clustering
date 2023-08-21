"""
This file contains some useful functions, e.g. function for loading the dataset, the reduced dataset etc..
"""

import pandas as pd

from tqdm import tqdm

def load_dataset():
    X = pd.read_parquet('dataset/X.parquet')
    y = pd.read_parquet('dataset/y.parquet').squeeze() 

    return X,y
    
def load_PCA_datasets(pca_dim:int):
    """
    This function loads the transformed datasets from the "dataset" folder.

    INPUT:
    - pca_dim, i.e. the PCA dimension.

    OUTPUT: ( {pca_dim : pca_dataset} , y )
    """
    
    y = pd.read_parquet('dataset/y.parquet').squeeze() 
    datasets = {}
    
    for i in tqdm(range(2,pca_dim+10,10), desc="Loading the PCA datasets.."):
        datasets[i] = pd.read_parquet("dataset/PCA_"+str(i)+".parquet")
        
    return datasets, y