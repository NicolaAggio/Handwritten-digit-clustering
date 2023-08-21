"""
This file contains the functions for tuning the hyperparameters of each of the clustering models.
"""
import pandas as pd
import time
import copy
import os
import pickle

from typing import Union, List, Tuple, Dict
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering
from sklearn.metrics.cluster import rand_score
from tqdm.notebook import tqdm
from utils import load_PCA_datasets

def score_calculation(model: Union[GaussianMixture, MeanShift, SpectralClustering], X:pd.DataFrame, y:pd.Series, hyperparameter_name:str, hyperparameter_val:Union[int,float]):
    """
    This function fits each of the clustering model with different values for the given hyperparamater, and finally calculates the rand score.

    INPUT:
    - model, i.e the model to use;
    - X, i.e. the feature vector;
    - y, i.e. the label vector;
    - hyperparameter_name, i.e. the hyperparameter name;
    - hyperparameter_val, i.e. the hyperparameter value.

    OUTPUT:
    - hyperparameter value (int|float);
    - number of clusters, only for the MeanShift algorithm (int);
    - rand score (float);
    """
    
    # fitting the model
    model = model.set_params(**{hyperparameter_name:hyperparameter_val})
    labels = model.fit_predict(X)
    score = rand_score(y, labels)
    
    if isinstance(model, MeanShift):
        return (hyperparameter_val, model.cluster_centers_.shape[0], score)
    else:
        return (hyperparameter_val, score)

def hyperparameter_tuning(desc:str, model: Union[GaussianMixture, MeanShift, SpectralClustering], hyperparameter_name:str, hyperparameter_values:List[Union[float, int]],  X:pd.DataFrame, y:pd.Series):
    """
    This function tunes the given hyperparameter for the given model.

    INPUT:
    - desc, i.e. a textual description of the current tuning;
    - model, i.e. the model for which the parameter is tuned;
    - hyperparameter_name, i.e. the name of the hyperparameter to tune;
    - hyperparameter_values, i.e. the list of hyperparameter values;
    - X, i.e. the feature vector;
    - y, i.e. the label vector;

    OUTPUT: 
    - result, i.e. a DataFrame containing, for each of the possible values of the hyperparameter to tune, the results of the trained model;
    - best_result_index, i.e. the index of the hyperparameter value which corresponds to the best rand score;
    - fitted_model, i.e. the model trained with the best hyperparamter value;
    - elapsed, i.e. the elapsded time for the training phase.
    """

    result = []
    fitted_model = None

    for val in tqdm(hyperparameter_values, desc=desc):
        result.append(score_calculation(model,X,y,hyperparameter_name,val))
    
    if isinstance(model, MeanShift):
        columns = [hyperparameter_name, 'n_clusters', 'rand_score']
    else:
        columns = [hyperparameter_name, 'rand_score']
        
    result = pd.DataFrame(result, columns=columns)
    best_rand_index = result.index[result["rand_score"]==result["rand_score"].max()].to_list()[0]

    start = time.time()
    
    fitted_model = model.set_params(**{hyperparameter_name:result[hyperparameter_name].iloc[best_rand_index].squeeze()}).fit(X)
    
    end = time.time()
    elapsed = end-start
    
    return result, best_rand_index, fitted_model, elapsed

def get_results(dfs:Dict[int, pd.DataFrame], y:pd.Series, model:Union[GaussianMixture, MeanShift, SpectralClustering], hyperparameter_name:str, hyperparameter_values:List[Union[float, int]]):
    """
    Function that given the PCA reduced data-frames and the responses dataframe,
    for each PCA dimension, tunes the given hyperparameter of the given model using the given hyperparameter values.

    Args:
        dfs (Dict[int, pd.DataFrame]): PCA reduced dataframes.
        y (pd.Series): response dataframe.
        model (Union[GaussianMixture, MeanShift, SpectralClustering]): model to tune.
        hyperparameter_name (str): hyperparameter to tune.
        hyperparameter_values (List[Union[float, int]]): list of hyperparameter values.

    Returns:tuple of:
                        -dict of:
                            -key: PCA dimension.
                            -value: result dataframe for this dimension.
                        
                        -dict of:
                            -key: PCA dimension.
                            -value: best result index for the result dataframe for this dimension.
                            
                        -dict of:
                            -key: PCA dimension.
                            -value: fitted model with the best hyperparameter for this dimension.
                            
                        -dict of:
                            -key: PCA dimension.
                            -value: fitting time for the best model for this dimension.
    """
    
    results = {}
    best_indexes = {}
    fitted_estimators = {}
    timings = {}
    
    for dim in tqdm(dfs.keys(),desc="Total result"):
        results[dim],best_indexes[dim],fitted_estimator,timings[dim]=hyperparameter_tuning("Tuning with PCA = "+str(dim)+"..",
                                                                                            model,
                                                                                            hyperparameter_name,
                                                                                            hyperparameter_values,
                                                                                            dfs[dim], 
                                                                                            y)
        
        fitted_estimators[dim]=copy.deepcopy(fitted_estimator)
        
    return results, best_indexes, fitted_estimators, timings

def save_results(model_name:str, results:Dict[int,pd.DataFrame], best_indexes:Dict[int,int], fitted_models:Dict[int,Union[GaussianMixture, MeanShift, SpectralClustering]], timings:Dict[int,float]):
    """
    Function that saves the results, the best indexes, fitted models and timings dicts for the given model.

    Args:
        model_name (str): model name.
        
        results (Dict[int,pd.DataFrame]):
            dict of:
                -key: PCA dimension.
                -value: result dataframe for this dimension.
                
        best_indexes (Dict[int,int]):
            dict of:
                -key: PCA dimension.
                -value: best result index for the result dataframe for this dimension.
                
        fitted_models (Dict[int,Union[GaussianMixture, MeanShift, SpectralClustering]]):
            dict of:
                -key: PCA dimension.
                -value: fitted model with the best hyperparameter for this dimension.
                
        timings (Dict[int,float]):
            dict of:
                -key: PCA dimension.
                -value: fitting time for the best model for this dimension.
    """
    
    PATH = os.getcwd()+"/"+model_name
    
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    
    with open(model_name+"/results.pkl", 'wb') as out:
        pickle.dump(results, out, pickle.HIGHEST_PROTOCOL)
    
    with open(model_name+"/best_indexes.pkl", 'wb') as out:
        pickle.dump(best_indexes, out, pickle.HIGHEST_PROTOCOL)
            
    with open(model_name+"/fitted_models.pkl", 'wb') as out:
        pickle.dump(fitted_models, out, pickle.HIGHEST_PROTOCOL)
    
    with open(model_name+"/timings.pkl", 'wb') as out:
        pickle.dump(timings, out, pickle.HIGHEST_PROTOCOL)

def tune_model(name:str, max_pca_dim:int):
    n_jobs = -1

    # loading the PCA transformed datasets
    datasets, y = load_PCA_datasets(max_pca_dim)

    match name:
        case "GaussianMixture":
            estimator = GaussianMixture(covariance_type = "diag", max_iter = 3000, random_state = 32)
            hyperparameter_name = "n_components"
            hyperparameter_values = [x for x in range(5,16)]
            
        case "MeanShift":
            estimator = MeanShift(n_jobs = n_jobs)
            hyperparameter_name = "bandwidth"
            hyperparameter_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 5, 10, 15, 20]
            
        case "NormalizedCut":
            estimator = SpectralClustering(affinity = "nearest_neighbors", n_neighbors = 40, n_jobs = n_jobs)
            hyperparameter_name = "n_clusters"
            hyperparameter_values = [x for x in range(5,16)]

        case _:
            print("Wrong model name...")
            exit(1)

    results,best_indexes,fitted_estimators,timings = get_results(datasets, y, estimator, hyperparameter_name, hyperparameter_values)

    return results, best_indexes, fitted_estimators, timings