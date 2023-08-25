import time
import numpy as np

from utils import load_PCA_test_sets
from typing import Dict, Union, List
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm

def best_PCA(tuning_results:Dict[int,tuple]):
    """
    This function retrieves the value of PCA for which the rand score on the validation set is maximum.

    INPUT: 
    - tuning_results = 
    """
    best_PCA = 0
    best_hyperparameters = {}
    max_rand_score = 0

    for pca_dim in tuning_results.keys():
        score = tuning_results[pca_dim][2]
        if score > max_rand_score:
            best_PCA = pca_dim
            best_hyperparameters = tuning_results[pca_dim][1]
            max_rand_score = score

    print("The best PCA dimension is " + str(best_PCA) + ", with hyperparameters = ", best_hyperparameters)

def get_training_times(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the training time of the model fitted with the best values of the hyperparameters.
    """
    training_times = []
    for pca_dim in tuning_results.keys():
        training_times.append(tuning_results[pca_dim][3])
    
    return training_times

def get_testing_times(testing_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the testing time of the model fitted with the best values of the hyperparameters.
    """
    testing_times = []
    for pca_dim in testing_results.keys():
        testing_times.append(testing_results[pca_dim][1])
    
    return testing_times

def get_training_rand_scores(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the rand scores of the model fitted with the best values of the hyperparameters.
    """
    rand_scores = []
    for pca_dim in tuning_results.keys():
        rand_scores.append(tuning_results[pca_dim][2])

    return rand_scores

def get_testing_rand_scores(testing_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the rand scores of the model fitted with the best values of the hyperparameters.
    """
    rand_scores = []
    for pca_dim in testing_results.keys():
        rand_scores.append(testing_results[pca_dim][0])

    return rand_scores

def get_n_clusters_tuning(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the number of clusters obtained by the MeanShift model with the best values of the hyperparameters.
    """
    n_clusters = []
    for pca_dim in tuning_results.keys():
        n_clusters.append(tuning_results[pca_dim][4])

    return n_clusters

def get_n_clusters_testing(testing_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the number of clusters obtained by the model with the best values of the hyperparameters.
    """
    n_clusters = []
    for pca_dim in testing_results.keys():
        n_clusters.append(testing_results[pca_dim][2])

    return n_clusters

def get_best_estimators(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the best estimator obtained in the tuning phase.
    """
    best_estimators = []
    for pca_dim in tuning_results.keys():
        best_estimators.append(tuning_results[pca_dim][0])

    return best_estimators

def get_labels(testing_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the labels obtained in the testing phase.
    """
    labels = []
    for pca_dim in testing_results.keys():
        labels.append(testing_results[pca_dim][3])

    return labels

def execute_estimators(best_estimators:List[Union[GaussianMixture, MeanShift, SpectralClustering]], model_name:str, max_pca_dim:int, dataset_percentage:float):
    """
    This function execute the best estimator on the test set.
    """
    results = {}
    i = 0

    X_test, y_test = load_PCA_test_sets(max_pca_dim, dataset_percentage)

    for dim in tqdm(X_test.keys(), "Executing " + model_name + " .."):
        best_estimator = best_estimators[i]

        start = time.time()
        labels = best_estimator.predict(X_test[dim])
        testing_time = time.time() - start
        rand = rand_score(labels, y_test)

        n_clusters = len(np.unique(labels))
        results[dim] = rand, testing_time, n_clusters, labels
        
        i += 1
    
    return results