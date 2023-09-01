"""
This file contains the functions for tuning the hyperparameters of each of the clustering models.
"""
import pandas as pd
import time
import os
import pickle

from sklearn.base import ClusterMixin, BaseEstimator
from typing import Dict, List
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.model_selection import GridSearchCV
from tqdm.notebook import tqdm
from loading import load_PCA_train_sets

class MySpectralClustring(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        *,
        eigen_solver=None,
        n_components=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol="auto",
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.eigen_solver = "lobpcg"
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        return SpectralClustering.fit(self,X,y)

    def fit_predict(self, X, y=None):
        return super().fit_predict(X, y)
    
    def predict(self, X, y=None):
        return self.fit_predict(X,y)

def get_results(X_train:Dict[int, pd.DataFrame], y_train:pd.Series, model:GridSearchCV, model_name:str):
    """
    This function performs the actual tuning of the hyperparameters for the specified model. We recall that the tuning is performed by using the GridSearchCV library.
    
    INPUT:
    - X_train, i.e. a dictionary {pca_dim : X_train}, for each value of PCA dimension;
    - y_train, i.e. the training label vector;
    - model, i.e. the model to be tuned;
    - model_name, i.e. the name of the model to be tuned.

    OUTPUT: a dictionary in the form {pca_dim : (best_estimator, best_params, best_rand_score, trainig time, n_clusters (only for MeanShift))}.
    """
    res = {}
    
    # recall: X_train.keys() = [pca_dim1, ..]
    for dim in tqdm(X_train.keys(), "Tuning " + model_name + " .."):
        start = time.time()
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

def tune_model(model_name:str, pca_dimensions:List[int], dataset_percentage:float):
    """
    This function sets the hyperparameters (names and values) for the provided model.

    INPUT:
    - model_name, i.e. the ame of the model to be tuned;
    - pca_dimensions, i.e. a list of all the possible PCA dimensions;
    - dataset_percentage, i.e. the percentage of dataset that is considered.

    OUTPUT: see get_results() function.
    """
    n_jobs = -1

    # loading the PCA transformed datasets
    X_train, y_train = load_PCA_train_sets(pca_dimensions, dataset_percentage)

    # setting the parameters according to the provided model
    match model_name:
        case "GaussianMixture":
            param_grid = {"n_components" : [x for x in range(5,16)],
            "init_params" : ["kmeans", "random_from_data"]}

            model = GridSearchCV(GaussianMixture(covariance_type="diag", random_state=1, max_iter=200), param_grid, scoring="rand_score", refit="rand_score", n_jobs=n_jobs, cv=5, error_score="raise")
            
        case "MeanShift":
            param_grid = {"bandwidth" : [0.6, 1, 3, 4, 5]}
            model = GridSearchCV(MeanShift(n_jobs=n_jobs), param_grid, scoring="rand_score", refit="rand_score", n_jobs=n_jobs, cv=5, error_score="raise")
            
        case "NormalizedCut":
            param_grid = {"n_clusters" : [x for x in range(5,16)],
            "assign_labels" : ["kmeans", "discretize", "cluster_qr"]}

            model = GridSearchCV(MySpectralClustring(affinity="nearest_neighbors", n_jobs=n_jobs, random_state=1), param_grid, scoring="rand_score", refit="rand_score", n_jobs=n_jobs, cv=5, error_score="raise")

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

    with open(os.getcwd() + "/GridSearch_tuning/" + model_name + '.pkl', "wb") as out:
        pickle.dump(result, out, pickle.HIGHEST_PROTOCOL)