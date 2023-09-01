"""
This file contains the functions for executing the best estimators obtained in the tuning phase and for retrieving the results of the training and testing phases.
"""

import time
import numpy as np

from loading import load_PCA_test_sets
from typing import Dict, Union, List
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.metrics.cluster import rand_score
from tqdm import tqdm
from GridSearch_tuning import MySpectralClustring

def best_PCA(tuning_results:Dict[int,tuple]):
    """
    This function retrieves the value of PCA for which the rand score on the validation set is maximum.

    INPUT: 
    - tuning_results = {pca_dim : (best_estimator, best_params, best_rand_score, trainig time, n_clusters (only for MeanShift))}.
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
    
    INPUT: 
    - tuning_results = {pca_dim : (best_estimator, best_params, best_rand_score, trainig time, n_clusters (only for MeanShift))}.
    
    OUTPUT:
    - training_times, i.e. the list of training_times for each PCA dimension.
    """
    training_times = []
    for pca_dim in tuning_results.keys():
        training_times.append(tuning_results[pca_dim][3])
    
    return training_times

def get_testing_times(testing_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the testing time of the model fitted with the best values of the hyperparameters.
    
    INPUT: 
    - testing_results = {pca_dim : (rand_score, testing_time, n_clusters, labels)}.
    
    OUTPUT:
    - testing_times, i.e. the list of testing_times for each PCA dimension.
    """
    testing_times = []
    for pca_dim in testing_results.keys():
        testing_times.append(testing_results[pca_dim][1])
    
    return testing_times

def get_training_rand_scores(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the rand scores of the model fitted with the best values of the hyperparameters.
    
    INPUT: 
    - tuning_results = {pca_dim : (best_estimator, best_params, best_rand_score, trainig time, n_clusters (only for MeanShift))}.
    
    OUTPUT:
    - rand_scores, i.e. the list of rand_scores for each PCA dimension.
    """
    rand_scores = []
    for pca_dim in tuning_results.keys():
        rand_scores.append(tuning_results[pca_dim][2])

    return rand_scores

def get_testing_rand_scores(testing_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the rand scores of the model fitted with the best values of the hyperparameters.
    
    INPUT: 
    - testing_results = {pca_dim : (rand_score, testing_time, n_clusters, labels)}.
    
    OUTPUT:
    - rand_scores, i.e. the list of rand_scores for each PCA dimension.
    """
    rand_scores = []
    for pca_dim in testing_results.keys():
        rand_scores.append(testing_results[pca_dim][0])

    return rand_scores

def get_n_clusters_tuning(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the number of clusters obtained by the MeanShift model with the best values of the hyperparameters.
    
    INPUT: 
    - tuning_results = {pca_dim : (best_estimator, best_params, best_rand_score, trainig time, n_clusters (only for MeanShift))}.
    
    OUTPUT:
    - n_clusters, i.e. the list of n_clusters for each PCA dimension. 
    """
    n_clusters = []
    for pca_dim in tuning_results.keys():
        n_clusters.append(tuning_results[pca_dim][4])

    return n_clusters

def get_n_clusters_testing(testing_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the number of clusters obtained by the model with the best values of the hyperparameters.
    
    INPUT: 
    - testing_results = {pca_dim : (rand_score, testing_time, n_clusters, labels)}.
    
    OUTPUT:
    - n_clusters, i.e. the list of n_clusters for each PCA dimension.
    """
    n_clusters = []
    for pca_dim in testing_results.keys():
        n_clusters.append(testing_results[pca_dim][2])

    return n_clusters

def get_best_estimators(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the best estimator obtained in the tuning phase.
    
    INPUT: 
    - tuning_results = {pca_dim : (best_estimator, best_params, best_rand_score, trainig time, n_clusters (only for MeanShift))}.
    
    OUTPUT:
    - labels, i.e. the list of best_estimators for each PCA dimension. 
    """
    best_estimators = []
    for pca_dim in tuning_results.keys():
        best_estimators.append(tuning_results[pca_dim][0])

    return best_estimators

def get_labels(testing_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the labels obtained in the testing phase.
    
    INPUT: 
    - testing_results = {pca_dim : (rand_score, testing_time, n_clusters, labels)}.

    OUTPUT:
    - labels, i.e. the list of labels for each PCA dimension.
    """
    labels = []
    for pca_dim in testing_results.keys():
        labels.append(testing_results[pca_dim][3])

    return labels

def execute_estimators(best_estimators:List[Union[GaussianMixture, MeanShift, MySpectralClustring]], model_name:str, pca_dimensions:List[int], dataset_percentage:float):
    """
    This function execute the best estimator on the test set.
    
    INPUT:
    - best_estimators, i.e. the best estimators for each PCA dimension;
    - model_name, i.e. the name of the model;
    - pca_dimensions, i.e. the list of all possible values of PCA dimension;
    - dataset_percentage, i.e. the percentage of dataset.

    OUTPUT:
    - results, i.e. {pca_dim : (rand_score, testing_time, n_clusters, labels)}.
    """
    results = {}
    i = 0

    X_test, y_test = load_PCA_test_sets(pca_dimensions, dataset_percentage)

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