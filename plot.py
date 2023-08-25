import matplotlib.pyplot as plt
import numpy as np
import random

from utils import load_PCA_2
from typing import List

def plot_digits(iter, X, y):
    fig, axs = plt.subplots(10, iter=15)
    
    for digit in range(10):
        for x in range(iter):
            digit_index = y[y == digit].index[x]
            digit_pixels = np.array(X.iloc[digit_index]).reshape(28, 28)
            axs[digit,x].imshow(digit_pixels)
            axs[digit,x].axis('off')

def plot_rand_score_vs_PCA(PCA_dimensions, rand_scores):
    """
    
    """
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(PCA_dimensions, rand_scores,'-*' , color = 'blue')
    ax.grid(True)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Rand score")
    ax.set_title('Value of the rand score vs PCA dimension', weight = 'bold')
    plt.show()
    
def plot_training_time_vs_PCA(PCA_dimensions, training_times):
    """
    
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(PCA_dimensions, training_times,'-*' , color = 'red')
    ax.grid(True)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Training time (s)")
    ax.set_title('Training times vs PCA dimension', weight = 'bold')
    plt.show()

def plot_testing_time_vs_PCA(PCA_dimensions, testing_times):
    """
    
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(PCA_dimensions, testing_times,'-*' , color = 'red')
    ax.grid(True)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Testing time (s)")
    ax.set_title('Testing times vs PCA dimension', weight = 'bold')
    plt.show()

def plot_clusters(dataset_percentage:float, n_clusters:int, labels:List):
    """
    This function plots the clusters obtained by the model with PCA dimension equal to 2.

    INPUT:
    - 
    """

    X, _ = load_PCA_2(dataset_percentage)
        
    plt.figure(figsize=(20,10))
    plt.clf()

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_clusters)]

    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
        
        plt.scatter(X[my_members]["pca_1"], X[my_members]["pca_2"], color=col) 
        
    plt.title("Number of clusters: %d" % n_clusters, weight = "bold")
    plt.show()