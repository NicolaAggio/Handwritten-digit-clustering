from typing import Dict

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

def get_rand_scores(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the rand scores of the model fitted with the best values of the hyperparameters.
    """
    rand_scores = []
    for pca_dim in tuning_results.keys():
        rand_scores.append(tuning_results[pca_dim][2])

    return rand_scores

def get_n_clusters(tuning_results:Dict[int,tuple]):
    """
    This function retrieves, for each PCA dimension value, the number of clusters obtained by the MeanShift model with the best values of the hyperparameters.
    """
    n_clusters = []
    for pca_dim in tuning_results.keys():
        n_clusters.append(tuning_results[pca_dim[4]])

    return get_n_clusters