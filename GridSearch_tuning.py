"""
This file contains the functions for tuning the hyperparameters of each of the clustering models.
"""
import pandas as pd
import time
import os
import pickle

from sklearn.base import ClusterMixin, BaseEstimator
from statistics import mean
from typing import Dict, List
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.model_selection import GridSearchCV, cross_val_score
from tqdm.notebook import tqdm
from utils import load_PCA_train_sets

class MySpectralClustring(ClusterMixin, BaseEstimator):
    # _parameter_constraints: dict = {
    #     "n_clusters": [Interval(Integral, 1, None, closed="left")],
    #     "eigen_solver": [StrOptions({"arpack", "lobpcg", "amg"}), None],
    #     "n_components": [Interval(Integral, 1, None, closed="left"), None],
    #     "random_state": ["random_state"],
    #     "n_init": [Interval(Integral, 1, None, closed="left")],
    #     "gamma": [Interval(Real, 0, None, closed="left")],
    #     "affinity": [
    #         callable,
    #         StrOptions(
    #             set(KERNEL_PARAMS)
    #             | {"nearest_neighbors", "precomputed", "precomputed_nearest_neighbors"}
    #         ),
    #     ],
    #     "n_neighbors": [Interval(Integral, 1, None, closed="left")],
    #     "eigen_tol": [
    #         Interval(Real, 0.0, None, closed="left"),
    #         StrOptions({"auto"}),
    #     ],
    #     "assign_labels": [StrOptions({"kmeans", "discretize", "cluster_qr"})],
    #     "degree": [Interval(Integral, 0, None, closed="left")],
    #     "coef0": [Interval(Real, None, None, closed="neither")],
    #     "kernel_params": [dict, None],
    #     "n_jobs": [Integral, None],
    #     "verbose": ["verbose"],
    # }

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

def tune_model(model_name:str, max_pca_dim:List[int], dataset_percentage:float):
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

    with open(model_name + '.pkl', "wb") as out:
        pickle.dump(result, out, pickle.HIGHEST_PROTOCOL)

def tune_SpectralClustering(max_pca_dim, dataset_percentage):
    results = {}
    X_train, y_train = load_PCA_train_sets(max_pca_dim, dataset_percentage)

    for dim in tqdm(X_train.keys(), "Tuning SpectralClustering" + " .."):
        results[dim] = {}
        for n_clusters in [x for x in range(5,16)]:
            model = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", n_jobs=-1, random_state=1)

            start_train = time.time()
            scores = cross_val_score(model, X_train[dim], y_train, cv=5, scoring="rand_score", n_jobs=-1)
            time_train = time.time() - start_train

            print("PCA dimension = ", dim)
            print("Number of clusters = ", n_clusters)
            print("Rand score = ", mean(scores))

            results[dim][n_clusters] = (mean(scores), time_train)

    return results