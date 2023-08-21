"""
This file contains the function for reducing the size of the MNIST dataset and for applying the PCA transformations to the dataset.
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from sklearn.decomposition import PCA

def plot_digits(iter, X, y):
    fig, axs = plt.subplots(10, iter=15)
    
    for digit in range(10):
        for x in range(iter):
            digit_index = y[y == digit].index[x]
            digit_pixels = np.array(X.iloc[digit_index]).reshape(28, 28)
            axs[digit,x].imshow(digit_pixels)
            axs[digit,x].axis('off')

def apply_PCA(X:pd.DataFrame, y:pd.Series, pca_dim:int, dataset_percentage:float):
    """
    This function applies some PCA transformations to a reduced fraction of the dataset.\nThe PCA dimension varies from 2 to 200.
    """
    
    # random sampling
    r = np.random.RandomState(1)
    indexes = r.choice(70000, int(70000*dataset_percentage),replace=False)
    
    # saving the sampled dataset 
    X.iloc[indexes].to_parquet("dataset/X.parquet")
    y[indexes].to_frame().to_parquet("dataset/y.parquet")
    
    for i in tqdm(range(2,pca_dim+10,10)):
        pca = PCA(n_components=i)
        
        # applying PCA to the sampled dataset
        df = pd.DataFrame(pca.fit_transform(X),columns=["PCA_"+str(x) for x in range(1,i+1)])
        df = df.iloc[indexes]
        
        # saving the reduced dataset 
        df.to_parquet("dataset/PCA_"+str(i)+".parquet")
        
        # with open("PCA/"+str(i)+".pkl", 'wb') as out:
        #     pickle.dump(pca, out, pickle.HIGHEST_PROTOCOL)