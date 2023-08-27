import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os

from utils import load_PCA_2
from typing import List, Dict

path = os.getcwd() + "/results/"

def plot_digits(iter, X, y):
    fig, axs = plt.subplots(10, iter=15)
    
    for digit in range(10):
        for x in range(iter):
            digit_index = y[y == digit].index[x]
            digit_pixels = np.array(X.iloc[digit_index]).reshape(28, 28)
            axs[digit,x].imshow(digit_pixels)
            axs[digit,x].axis('off')

def plot_rand_score_vs_PCA(PCA_dimensions:List[int], rand_scores:List[float], label:str, model_name:str):
    """
    
    """
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(PCA_dimensions, rand_scores,'-*' , color = 'blue')
    ax.grid(True)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Rand score")
    ax.set_title('Value of the rand score vs PCA dimension', weight = 'bold')
    plt.show()
    fig.savefig(path + model_name + "/rand_vs_PCA_" + label + ".png")
    
def plot_training_time_vs_PCA(PCA_dimensions:List[int], training_times:List[float], model_name:str):
    """
    
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(PCA_dimensions, training_times,'-*' , color = 'red')
    ax.grid(True)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Training time (s)")
    ax.set_title('Training times vs PCA dimension', weight = 'bold')
    plt.show()
    fig.savefig(path + model_name + "/training_time_vs_PCA.png")

def plot_testing_time_vs_PCA(PCA_dimensions:List[int], testing_times:List[float], model_name:str):
    """
    
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(PCA_dimensions, testing_times,'-*' , color = 'red')
    ax.grid(True)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Testing time (s)")
    ax.set_title('Testing times vs PCA dimension', weight = 'bold')
    plt.show()
    fig.savefig(path + model_name + "/testing_time_vs_PCA.png")

def plot_clusters(dataset_percentage:float, n_clusters:int, labels:List[int], model_name:str):
    """
    This function plots the clusters obtained by the model with PCA dimension equal to 2.

    INPUT:
    - 
    """

    X, _ = load_PCA_2(dataset_percentage)
        
    fig = plt.figure(figsize=(20,10))
    plt.clf()

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_clusters)]

    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
        
        plt.scatter(X[my_members]["pca_1"], X[my_members]["pca_2"], color=col) 
        
    plt.title("Number of clusters: %d" % n_clusters, weight = "bold")
    plt.show()
    fig.savefig(path + model_name + "/clusters.png")

def plot_images_per_cluster(X_test:Dict, pca_dim:int, labels:List, n_clusters:int, model_name:str):
    """
    This function plots 4 images of each of the obtained clusters, given the PCA dimensions.
    """

    fig,axs = plt.subplots(n_clusters, 4, figsize = (4*2, n_clusters*2))
    fig.suptitle(f"PCA dimension: {pca_dim}", weight = "bold")

    df = X_test[pca_dim]

    for item in [item for sublist in axs for item in sublist]:
        item.set_yticklabels([])
        item.set_xticklabels([])
                
    for k in range(n_clusters):
        my_members = labels == k
        with open("PCA/"+str(pca_dim)+".pkl", 'rb') as inp:
            pca = pickle.load(inp)
        data = pca.inverse_transform(df[my_members])
    
        if data.shape[0]>=4:
            random_indexes=np.random.choice(data.shape[0], size=4, replace=False)
        else:
            random_indexes=np.random.choice(data.shape[0], size=data.shape[0], replace=False)
            
        for i,imag in enumerate(data[random_indexes, :]):
                axs[k,i].imshow(imag.reshape(28, 28))
                
        if data.shape[0]==0:
            i=-1
    
        for j in range(i+1,4):
            axs[k,j].imshow(np.zeros(28*28).reshape(28,28))
            
    for ax,c in zip(axs[:,0],range(n_clusters)):
        ax.set_ylabel(str(c), rotation=0, size='large')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.show()
    fig.savefig(path + model_name + "/images_per_cluster.png")