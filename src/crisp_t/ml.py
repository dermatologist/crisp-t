import numpy
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KDTree
from random import randint
import logging

from .model import Corpus
from .csv import Csv

logger = logging.getLogger(__name__)
ML_INSTALLED = False

try:
    from xgboost import XGBClassifier
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules

    import torch.nn as nn
    import torch.optim as optim
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from imblearn.over_sampling import RandomOverSampler

    ML_INSTALLED = True

    class NeuralNet(nn.Module):
        def __init__(self, input_dim):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, 12)
            self.fc2 = nn.Linear(12, 8)
            self.fc3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

except ImportError:
    logger.info(
        "ML dependencies are not installed. Please install them by ```pip install qrmine[ml] to use ML features."
    )


class ML:
    def __init__(
        self,
        corpus: Corpus,
    ):
        if not ML_INSTALLED:
            raise ImportError("ML dependencies are not installed.")
        self._corpus = corpus
        self._epochs = 1
        self._samplesize = 0
        self._csv = None

    @property
    def csv(self):
        return self._csv

    @csv.setter
    def csv(self, value):
        if isinstance(value, Csv):
            self._csv = value
        else:
            raise ValueError("Value must be an instance of Csv class.")

    def get_kmeans(self, number_of_clusters=3, seed=42, verbose=True):
        if self._csv is None:
            raise ValueError(
                "CSV data is not set. Please set self.csv before calling get_kmeans."
            )
        X, _ = self._csv.read_xy("", ignore_columns=True, numeric_only=True)
        if X is None:
            raise ValueError(
                "Input features X are None. Cannot perform KMeans clustering."
            )
        kmeans = KMeans(
            n_clusters=number_of_clusters, init="k-means++", random_state=seed
        )
        self._clusters = kmeans.fit_predict(X)
        members = self.get_members(self._clusters, number_of_clusters)
        return self._clusters, members

    def get_members(self, clusters, number_of_clusters=3):
        members = []
        for i in range(number_of_clusters):
            members.append([])
        for i, cluster in enumerate(clusters):
            members[cluster].append(i)
        return members

    def profile(self, members, number_of_clusters=3):
        if self._csv is None:
            raise ValueError(
                "CSV data is not set. Please set self.csv before calling profile."
            )
        for i in range(number_of_clusters):
            print("Cluster: ", i)
            print("Cluster Length: ", len(members[i]))
            print("Cluster Members")
            if self._csv is not None and getattr(self._csv, "df", None) is not None:
                print(self._csv.df.iloc[members[i], :])
                print("Mean")
                print(self._csv.df.iloc[members[i], :].mean(axis=0))
            else:
                print("DataFrame (self._csv.df) is not set.")
        return members

    # def get_centroids(self, number_of_clusters=3, verbose=True):
    #     cluster_list = []
    #     for x in range(0, number_of_clusters):
    #         ct = 0
    #         for cluster in self._clusters:
    #             if cluster == x:
    #                 cluster_list.append(ct)
    #             ct += 1
    #         if verbose:
    #             print("Cluster: ", x)
    #             print("Cluster Length: ", len(cluster_list))
    #             print("Cluster Members")
    #             if self._csv is not None and getattr(self._csv, "df", None) is not None:
    #                 print(self._csv.df.iloc[cluster_list, :])
    #                 print("Mean")
    #                 print(self._csv.df.iloc[cluster_list, :].mean(axis=0))
    #             else:
    #                 print("DataFrame (self._csv.df) is not set.")
    #     return cluster_list
