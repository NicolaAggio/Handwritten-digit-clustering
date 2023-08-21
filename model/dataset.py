""" This file provides the class that represents a Dataset """

import pandas as pd
import numpy as np

from os import path
from utils import makedir, get_dataset_dir
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import rand_score

class Dataset:
    """
    This class represents a Dataset as a tuple:
     - _X = the feature matrix as a pandas dataframe;
     - _y = the label vector as an array.
    """

    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        """
        INPUT:
        - x: feature matrix;
        - y: label vector.
        """

        # data and labels must have the same length
        if len(x) != len(y):
            raise Exception(f"X has length {len(x)}, while y has length {len(y)}")

        self._X: pd.DataFrame = x
        self._y: np.ndarray = np.array(y)

    def X(self):
        """
        :return: feature matrix
        """
        return self._X

    def y(self):
        """
        :return: labels
        """
        return self._y

    def features(self):
        """
        :return: features name
        """
        return list(self.X.columns)

    def __len__(self):
        """
        :return: rows in the feature matrix
        """
        return len(self.X)

    def __str__(self):
        """
        :return: string containing the information about the dataset
        """
        return f"[Length: {len(self)}; Features: {len(self.X.columns)}]"

    # Dataset transformations and reductions

    def apply_pca(self, n_components: int):
        """
        Applies PCA to the feature space
        :param n_components: number of components for the PCA method
        :return: dataset with reduced number of components
        """

        # integrity checks
        if n_components < 0:
            raise Exception("The number of components must be a positive number")

        actual_components = len(self.X.columns)

        if n_components >= actual_components:
            raise Exception(f"The number of components must be less than {actual_components}, the actual number of components")

        # return new object
        new_x = pd.DataFrame(PCA(n_components=n_components).fit_transform(self.X))
        return Dataset(x=new_x, y=self.y)

    def reduce_to_percentage(self, percentage: float = 1.):
        """
        Returns a randomly reduced-percentage dataset
        return: new dataset
        """

        if not 0. <= percentage <= 1.:
            raise Exception(f"Percentage {percentage} not in range [0, 1] ")

        _, X, _, y = train_test_split(self.X, self.y, test_size=percentage)

        return Dataset(X, y)

    def rescale(self):
        """
        Rescales rows and columns in interval [0, 1]
        """
        new_X = pd.DataFrame(MinMaxScaler().fit_transform(self.X), columns=self.features)
        return Dataset(x=new_X,y=self.y)

    # Storing the dataset

    def store(self, x_name: str = 'dataX', y_name: str = 'datay'):
        """
        Stores the dataset in datasets directory
        :param x_name: name of feature file
        :param y_name: name of labels file
        """

        makedir(get_dataset_dir())

        x_out = path.join(get_dataset_dir(), f"{x_name}.csv")
        y_out = path.join(get_dataset_dir(), f"{y_name}.csv")

        print(f"Saving {x_out}..")
        self.X.to_csv(x_out, index=False)

        print(f"Saving {y_out}..")
        pd.DataFrame(self.y).to_csv(y_out, index=False)

    # Plotting the digits of the dataset

    def plot_digit_distribution(self, save: bool = False, file_name: str = 'histo.png'):
        """
        STUB for util function
        :param save: true to save the image
        :param file_name: file name if image is saved
        """
        digits_histogram(labels=self.y, save=save, file_name=file_name)

    def plot_digits(self):
        """
        Plots all digits in the dataset
        """
        for i in range(len(self)):
            pixels = np.array(self.X.iloc[i])
            plot_digit(pixels=pixels)

    def plot_mean_digit(self):
        """
        Plots mean of all digits in the dataset
        """
        plot_mean_digit(X=self.X)
