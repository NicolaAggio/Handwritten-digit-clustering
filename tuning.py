"""
This file contains the functions for tuning the hyperparameters of each of the clustering models.
"""
import pandas as pd
import time
import pickle
import numpy as np

from sklearn.base import ClusterMixin, BaseEstimator
from typing import Union, List, Dict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.metrics.cluster import rand_score
from tqdm.notebook import tqdm
from utils import load_PCA_train_sets


def evaluate_model(model: Union[GaussianMixture, MeanShift, SpectralClustering], X_valid:pd.DataFrame, y_valid:pd.DataFrame, hyperparameter_name:str, hyperparameter_val:Union[int,float]):
    """
    This function fits the given clustering model with different values for the given hyperparamater, and finally calculates the rand score.

    INPUT:
    - model, i.e the model to use;
    - X_train, i.e. the training feature vector;
    - X_valid, i.e. the validation feature vector;
    - y_valid, i.e. the validation label vector;
    - hyperparameter_name, i.e. the hyperparameter name;
    - hyperparameter_val, i.e. the hyperparameter value.

    OUTPUT:
    - hyperparameter value;
    - number of clusters (only for the MeanShift algorithm);
    - rand score;
    """
    
    # fitting the model
    model = model.set_params(**{hyperparameter_name:hyperparameter_val})
    labels = model.fit_predict(X_valid)
    score = rand_score(y_valid, labels)
    
    if isinstance(model, MeanShift):
        return (hyperparameter_val, model.cluster_centers_.shape[0], score)
    else:
        return (hyperparameter_val, score)

def tune_hyperparameter(desc:str, model: Union[GaussianMixture, MeanShift, SpectralClustering], hyperparameter_name:str, hyperparameter_values:List[Union[float, int]],  X_train:pd.DataFrame, X_valid:pd.DataFrame, y_valid:pd.Series):
    """
    This function tunes the given hyperparameter for the given model.

    INPUT:
    - desc, i.e. a textual description of the current tuning;
    - model, i.e. the model for which the parameter is tuned;
    - hyperparameter_name, i.e. the name of the hyperparameter to tune;
    - hyperparameter_values, i.e. the list of hyperparameter values;
    - X_train, i.e. the training feature vector;
    - X_valid, i.e. the validation feature vector;
    - y_valid, i.e. the validation label vector;

    OUTPUT: 
    - result, i.e. a DataFrame containing, for each of the possible values of the hyperparameter to tune, the results of the trained model (number of clusters and rand score);
    - best_index, i.e. the index of the hyperparameter value which corresponds to the best rand score;
    - fitted_model, i.e. the model trained with the best hyperparamter value;
    - training_time, i.e. the training time.
    """

    result = []
    fitted_model = None

    # evaluating the model with each value of the hyperparameter
    for val in tqdm(hyperparameter_values, desc=desc):
        result.append(evaluate_model(model, X_valid, y_valid, hyperparameter_name, val))
    
    if isinstance(model, MeanShift):
        columns = [hyperparameter_name, 'n_clusters', 'rand_score']
    else:
        columns = [hyperparameter_name, 'rand_score']
        
    result = pd.DataFrame(result, columns=columns)

    # selecting the index of the value for which the rand score is maximum
    best_index = result.index[result["rand_score"]==result["rand_score"].max()].to_list()[0]

    # fitting the model
    start = time.time()
    fitted_model = model.set_params(**{hyperparameter_name:result[hyperparameter_name].iloc[best_index].squeeze()}).fit(X_train)
    training_time = time.time() - start
    
    return result, best_index, fitted_model, training_time

def get_results(datasets_train:Dict[int, pd.DataFrame], datasets_valid:Dict[int, pd.DataFrame], y_valid:pd.Series, model:Union[GaussianMixture, MeanShift, SpectralClustering], hyperparameter_name:str, hyperparameter_values:List[Union[float, int]]):
    """
    This function tunes the specified hyperparameter of the model with the given value.

    INPUT:
    - datasets_train = {PCA_dim, dataset}, i.e. the PCA train transformed datasets;
    - datasets_valid = {PCA_dim, dataset}, i.e. the PCA validation transformed datasets;
    - y_valid, i.e. the validation label vector;
    - model, i.e. the model for which the tuning is performed;
    - hyperparameter_name, i.e. the hyperparameter to tune.
    - hyperparameter_values, i.e. the list of all the possible values of the specified parameter.

    OUTPUT: {PCA_dim : (best_index, fitted_model, training_time)}, where:
    - result is a DataFrame containing, for each of the possible values of the hyperparameter to tune, the results of the trained model (number of clusters and rand score);
    - best_index represents the index of the hyperparameter value which corresponds to the best rand score;
    - fitted_model represents the model trained with the best hyperparamter value;
    - training_time represents the training time.

    - results = {PCA dimension, result dataframe for this dimension};
    - best_indexes = {PCA dimension, best result index for the result dataframe for this dimension};
    - fitted_estimator = {PCA dimension, fitted model with the best hyperparameter for this dimension};
    - timings = {PCA dimension, fitting time for the best model for this dimension}.
    """
    
    res = {int:tuple}
    
    # recall: datasets_train.keys() = [pca_dim1, ..]
    for dim in tqdm(datasets_train.keys()):
        res[dim] = tune_hyperparameter("Tuning " + hyperparameter_name + " with PCA = " + str(dim) + "..",
                                                                                            model,
                                                                                            hyperparameter_name,
                                                                                            hyperparameter_values,
                                                                                            datasets_train[dim],
                                                                                            datasets_valid[dim], 
                                                                                            y_valid)
        
    return res

def tune_model(model:str, max_pca_dim:int, dataset_percentage:float):
    """
    This function tunes the hyperparameter of the specified model.

    INPUT:
    - model, i.e. the model for which the tuning is performed;
    - max_pca_dim, i.e. the maximum value of PCA.

    OUTPUT:{PCA_dim : (best_index, fitted_model, training_time)}, where:
    - result is a DataFrame containing, for each of the possible values of the hyperparameter to tune, the results of the trained model (number of clusters and rand score);
    - best_index represents the index of the hyperparameter value which corresponds to the best rand score;
    - fitted_model represents the model trained with the best hyperparamter value;
    - training_time represents the training time.


    - results = {PCA dimension, result dataframe for this dimension};
    - best_indexes = {PCA dimension, best result index for the result dataframe for this dimension};
    - fitted_estimator = {PCA dimension, fitted model with the best hyperparameter for this dimension};
    - timings = {PCA dimension, fitting time for the best model for this dimension}.
    """
    n_jobs = -1

    # loading the PCA transformed datasets
    datasets_train, y_train = load_PCA_train_sets(max_pca_dim, dataset_percentage)

    # setting the parameters according to the provided model
    match model:
        case "GaussianMixture":
            model = GaussianMixture(covariance_type = "diag", max_iter = 3000, random_state = 32)
            hyperparameter_name = "n_components"
            hyperparameter_values = [x for x in range(5,16)]
            
        case "MeanShift":
            model = MeanShift(n_jobs = n_jobs)
            hyperparameter_name = "bandwidth"
            hyperparameter_values = [0.6, 1, 3, 4, 5, 8, 10, 15, 20]
            
        case "NormalizedCut":
            model = MySpectralClustring(affinity = "nearest_neighbors", n_jobs = n_jobs)
            hyperparameter_name = "n_clusters"
            hyperparameter_values = [x for x in range(5,16)]

    return get_results(datasets_train, model, hyperparameter_name, hyperparameter_values)

def save_tuning_results(model_name:str, result:Dict[int,tuple]):
    """
    This function saves the results of the execution of the tune_model function for a given model into a dedicated folder.

    INPUT:
    - model_name, i.e. the model name;
    - result, i.e. {PCA_dim : (best_index, fitted_model, training_time), for each PCA_dim}.
    """

    with open("tuning_" + model_name + ".pkl", "wb") as out:
        pickle.dump(result, out, pickle.HIGHEST_PROTOCOL)