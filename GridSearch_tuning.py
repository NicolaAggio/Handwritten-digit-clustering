"""
This file contains the functions for tuning the hyperparameters of each of the clustering models.
"""
import pandas as pd
import time
import os
import pickle

from typing import Dict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.model_selection import GridSearchCV
from tqdm.notebook import tqdm
from utils import load_PCA_train_sets

def get_results(X_train:Dict[int, pd.DataFrame], y_train:pd.Series, model:GridSearchCV, model_name:str):
    """
    """
    res = {}
    
    # recall: X_train.keys() = [pca_dim1, ..]
    for dim in tqdm(X_train.keys(), "Tuning " + model_name + " .."):
        start = time.time()
        if model_name == "NormalizedCut":
            model.fit_predict(X_train[dim], y_train)
        else:
            model.fit(X_train[dim], y_train)
        training_time = time.time() - start

        best_estimator = model.best_estimator_
        best_rand_score = model.best_score_
        best_params = model.best_params_

        print("PCA dimension = ", dim)
        print("Best rand score = ", best_rand_score)
        print("Best parameters = ", best_params)

        if (model_name == "MeanShift"):
            n_clusters = best_estimator.cluster_centers_.shape[0]
            print("Number of clusters = ", n_clusters)
            res[dim] = best_estimator, best_params, best_rand_score, training_time, n_clusters

        else:
            res[dim] = best_estimator, best_params, best_rand_score, training_time

    return res

def tune_model(model_name:str, max_pca_dim:int, dataset_percentage:float):
    """
    This function sets the hyperparameters (names and values) for the provided model.

    INPUT:
    - 
    """
    n_jobs = -1

    # loading the PCA transformed datasets
    X_train, y_train = load_PCA_train_sets(max_pca_dim, dataset_percentage)

    # setting the parameters according to the provided model
    match model_name:
        case "GaussianMixture":
            param_grid = {"n_components" : [x for x in range(5,16)],
            "init_params" : ["kmeans", "random_from_data"]}

            model = GridSearchCV(GaussianMixture(covariance_type="diag", random_state=1, max_iter=200), param_grid, scoring="rand_score", refit="rand_score", n_jobs=n_jobs, cv=5, error_score="raise")
            
        case "MeanShift":
            param_grid = {"bandwidth" : [0.6, 1, 3, 4, 5, 8]}
            model = GridSearchCV(MeanShift(n_jobs=n_jobs), param_grid, scoring="rand_score", refit="rand_score", n_jobs=n_jobs, cv=5, error_score="raise")
            
        case "NormalizedCut":
            param_grid = {"n_clusters" : [x for x in range(5,16)],
            "n_neighbors" : [10, 20, 30, 40],
            "assign_labels" : ["kmeans", "discretize", "cluster_qr"]}

            model = GridSearchCV(SpectralClustering(affinity="nearest_neighbors", n_jobs=n_jobs, random_state=1), param_grid, scoring="rand_score", refit="rand_score", n_jobs=n_jobs, cv=5, error_score="raise")

        case _:
            print("Wrong model name...")
            exit(1)

    return get_results(X_train, y_train, model, model_name)

def save_results(model_name:str, result:Dict[int,tuple]):
    """
    This function saves the results of the execution of the tune_model function for a given model into a dedicated folder.

    INPUT:
    - model_name, i.e. the model name;
    - result, i.e. {PCA_dim : (best_index, fitted_model, training_time), for each PCA_dim}.
    """
    if not os.path.exists(os.getcwd() + "/GridSearch_tuning"):
        os.mkdir(os.getcwd() + "/GridSearch_tuning")

    with open(model_name + '.pkl', "wb") as out:
        pickle.dump(result, out, pickle.HIGHEST_PROTOCOL)