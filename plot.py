import matplotlib.pyplot as plt
import numpy as np

def plot_digits(iter, X, y):
    fig, axs = plt.subplots(10, iter=15)
    
    for digit in range(10):
        for x in range(iter):
            digit_index = y[y == digit].index[x]
            digit_pixels = np.array(X.iloc[digit_index]).reshape(28, 28)
            axs[digit,x].imshow(digit_pixels)
            axs[digit,x].axis('off')

def plot_rand_score_vs_PCA(PCA_dimensions, rand_scores):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(PCA_dimensions, rand_scores,'-*' , color = 'blue')
    ax.grid(True)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Rand score")
    ax.set_title('Value of the rand score vs PCA dimension', weight = 'bold')
    plt.show()
    
def plot_training_time_vs_PCA(PCA_dimensions, training_times):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(PCA_dimensions, training_times,'-*' , color = 'red')
    ax.grid(True)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Training times")
    ax.set_title('Training times vs PCA dimension', weight = 'bold')
    plt.show()