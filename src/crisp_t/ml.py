import logging
from random import randint

import numpy
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .csv import Csv

logger = logging.getLogger(__name__)
ML_INSTALLED = False
torch = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from imblearn.over_sampling import RandomOverSampler
    from mlxtend.frequent_patterns import apriori, association_rules
    from torch.utils.data import DataLoader, TensorDataset
    from xgboost import XGBClassifier

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
        "ML dependencies are not installed. Please install them by ```pip install crisp-t[ml] to use ML features."
    )


class ML:
    def __init__(
        self,
        csv: Csv,
    ):
        if not ML_INSTALLED:
            raise ImportError("ML dependencies are not installed.")
        self._csv = csv
        self._epochs = 1
        self._samplesize = 0

    @property
    def csv(self):
        return self._csv

    @property
    def corpus(self):
        return self._csv.corpus

    @csv.setter
    def csv(self, value):
        if isinstance(value, Csv):
            self._csv = value
        else:
            raise ValueError(f"The input belongs to {type(value)} instead of Csv.")

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
        _df = self._csv.df
        # Create a column called numeric_cluster and assign cluster labels
        _df["numeric_cluster"] = clusters
        self._csv.df = _df
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
        _corpus = self._csv.corpus
        _numeric_clusters = ""
        for i in range(number_of_clusters):
            print("Cluster: ", i)
            print("Cluster Length: ", len(members[i]))
            print("Cluster Members")
            if self._csv is not None and getattr(self._csv, "df", None) is not None:
                print(self._csv.df.iloc[members[i], :])
                print("Centroids")
                print(self._csv.df.iloc[members[i], :].mean(axis=0))
                _numeric_clusters += f"Cluster {i} with {len(members[i])} members\n has the following centroids (mean values):\n"
                _numeric_clusters += f"{self._csv.df.iloc[members[i], :].mean(axis=0)}\n"
            else:
                print("DataFrame (self._csv.df) is not set.")
        if _corpus is not None:
            _corpus.metadata["numeric_clusters"] = _numeric_clusters
            self._csv.corpus = _corpus
        return members

    # ...existing code...
    def get_binary_nnet_predictions(self, y: str):
        if ML_INSTALLED is False:
            logger.info(
                "ML dependencies are not installed. Please install them by ```pip install crisp-t[ml] to use ML features."
            )
            return None

        # Prepare data (X features, Y target)
        X, Y = self._csv.prepare_data(y=y, oversample=False)
        if X is None or Y is None:
            raise ValueError("prepare_data returned None for X or Y.")

        # Ensure numpy float32 arrays (avoid torch tensor creation error on DataFrame)
        if hasattr(X, "to_numpy"):
            X_np = X.to_numpy(dtype=numpy.float32)
        else:
            X_np = numpy.asarray(X, dtype=numpy.float32)
        if hasattr(Y, "to_numpy"):
            Y_np = Y.to_numpy(dtype=numpy.float32)
        else:
            Y_np = numpy.asarray(Y, dtype=numpy.float32)

        # --- NEW: Normalize target for BCELoss to be strictly binary {0,1} ---
        unique_classes = numpy.unique(Y_np)
        if unique_classes.size != 2:
            raise ValueError(
                f"BCELoss requires binary targets, but found {unique_classes.size} classes: {unique_classes}. "
                "Provide a binary target column or extend the code to handle multi-class with CrossEntropyLoss."
            )

        mapping_applied = False
        class_mapping = {}
        inverse_mapping = {}
        # If classes are not already {0.0,1.0}, map them deterministically
        if not numpy.array_equal(
            numpy.sort(unique_classes), numpy.array([0.0, 1.0], dtype=numpy.float32)
        ):
            sorted_classes = sorted(unique_classes.tolist())
            class_mapping = {sorted_classes[0]: 0.0, sorted_classes[1]: 1.0}
            inverse_mapping = {v: k for k, v in class_mapping.items()}
            Y_np = numpy.vectorize(class_mapping.get)(Y_np).astype(numpy.float32)
            mapping_applied = True
            logger.info(
                f"Mapped original target classes {sorted_classes} to [0.0, 1.0] for BCELoss."
            )

        vnum = X_np.shape[1]

        model = NeuralNet(vnum)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Convert data to PyTorch tensors
        X_tensor = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(Y_np).view(-1, 1)

        # Create a dataset and data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        # Train the model
        for epoch in range(self._epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                if torch.isnan(loss):
                    raise RuntimeError("Encountered NaN loss during training.")
                loss.backward()
                optimizer.step()

        # Inference
        with torch.no_grad():
            raw_outputs = model(torch.from_numpy(X_np)).view(-1, 1)
            preds_np = raw_outputs.cpu().numpy().flatten()
            binary_preds = (preds_np >= 0.5).astype(int)

        # Map back to original class labels if remapped
        if mapping_applied:
            binary_preds = [inverse_mapping[int(p)] for p in binary_preds]
        else:
            binary_preds = binary_preds.tolist()

        # Calculate accuracy (compare in binary space)
        if mapping_applied:
            # Need original Y in same label space as predictions
            Y_eval = numpy.vectorize(class_mapping.get)(
                Y_np_original := numpy.asarray(
                    Y.to_numpy() if hasattr(Y, "to_numpy") else Y, dtype=numpy.float32
                )
            ).astype(int)
            preds_for_acc = numpy.vectorize(class_mapping.get)(
                numpy.asarray(binary_preds)
            ).astype(int)
        else:
            Y_eval = Y_np.astype(int)
            preds_for_acc = numpy.asarray(binary_preds).astype(int)

        correct = int((preds_for_acc == Y_eval).sum())
        total = len(preds_for_acc)
        accuracy = correct / total if total else 0.0
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return binary_preds
