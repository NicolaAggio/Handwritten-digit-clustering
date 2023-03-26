import numpy as np
import matplotlib.pyplot as plt
import random

def correctly_clustered(labels, clusters):
    count = 0
    for i in range(len(labels) - 1):
        for j in range(i+1, len(labels)):
            if i != j:
                if labels[i] == labels[j] and clusters[i] == clusters[j]:
                    count += 1
    return count

def correctly_unclustered(labels, clusters):
    count = 0
    for i in range(len(labels) - 1):
        for j in range(i+1, len(labels)):
            if i != j:
                if labels[i] != labels[j] and clusters[i] != clusters[j]:
                    count += 1
    return count

def rand_index(labels, clusters):
    a = correctly_clustered(labels, clusters)
    b = correctly_unclustered(labels, clusters)
    n = len(labels)
    return 2*(a+b)/(n*(n-1))

def plot_clustering(X, labels, cluster_centers=None):
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    plt.figure(figsize=(20,10))
    plt.clf()

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_clusters_)]
    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        
        plt.scatter(X[my_members]["PC_1"], X[my_members]["PC_2"], marker=markers[k%len(markers)], color=col)
        
        if cluster_centers is not None:
            cluster_center = cluster_centers[k]
            plt.scatter(cluster_center[0],cluster_center[1],marker=markers[k%len(markers)], edgecolor="black", s=150, color =col)
        
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()


# y = [1,1,1,0,0,0]
# labels = [1,1,0,0,0,0]

# print(correctly_clustered(y, labels))
# print(correctly_unclustered(y, labels))
# print(rand_index(y, labels))