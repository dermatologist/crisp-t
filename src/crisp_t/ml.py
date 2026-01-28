import logging
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tabulate import tabulate
from tqdm import tqdm

from .csv import Csv
from .mlib import config as ml_config

warnings.filterwarnings("ignore", category=DeprecationWarning)

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

    class MultiClassNet(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)  # raw logits for CrossEntropyLoss
            return x

    class SimpleLSTM(nn.Module):
        def __init__(
            self,
            vocab_size,
            embedding_dim=ml_config.LSTM_EMBEDDING_DIM,
            hidden_dim=ml_config.LSTM_HIDDEN_DIM,
            output_dim=1,
            num_layers=2,
            bidirectional=True,
            dropout=0.5,
        ):
            super(SimpleLSTM, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.fc = nn.Linear(lstm_output_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.sigmoid = nn.Sigmoid()

        def forward(self, text):
            embedded = self.dropout(self.embedding(text))
            lstm_out, (hidden, cell) = self.lstm(embedded)
            # Use the final hidden state from both directions
            if self.lstm.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                hidden = hidden[-1, :, :]
            hidden = self.dropout(hidden)
            output = self.fc(hidden)
            return self.sigmoid(output)

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
        self._epochs = 3
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

    def get_kmeans(self, number_of_clusters=3, seed=42, verbose=True, mcp=False):
        if self._csv is None:
            raise ValueError(
                "CSV data is not set. Please set self.csv before calling get_kmeans."
            )
        X, _ = self._csv.read_xy("")  # No output variable for clustering
        if X is None:
            raise ValueError(
                "Input features X are None. Cannot perform KMeans clustering."
            )
        kmeans = KMeans(
            n_clusters=number_of_clusters, init="k-means++", random_state=seed
        )
        self._clusters = kmeans.fit_predict(X)
        members = self._get_members(self._clusters, number_of_clusters)
        # Add cluster info to csv to metadata_cluster column
        if self._csv is not None and getattr(self._csv, "df", None) is not None:
            self._csv.df["metadata_cluster"] = self._clusters
        if verbose:
            print("KMeans Cluster Centers:\n", kmeans.cluster_centers_)
            print(
                "KMeans Inertia (Sum of squared distances to closest cluster center):\n",
                kmeans.inertia_,
            )
            if self._csv.corpus is not None:
                self._csv.corpus.metadata[ml_config.METADATA_KEY_KMEANS] = (
                    f"KMeans clustering with {number_of_clusters} clusters. Inertia: {kmeans.inertia_}"
                )
        # Add members info to corpus metadata
        members_info = "\n".join(
            [
                f"Cluster {i}: {len(members[i])} members"
                for i in range(number_of_clusters)
            ]
        )
        if self._csv.corpus is not None:
            self._csv.corpus.metadata["kmeans_members"] = (
                f"KMeans clustering members:\n{members_info}"
            )
        if mcp:
            return members_info
        return self._clusters, members

    def _get_members(self, clusters, number_of_clusters=3):
        _df = self._csv.df
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
                _numeric_clusters += (
                    f"{self._csv.df.iloc[members[i], :].mean(axis=0)}\n"
                )
            else:
                print("DataFrame (self._csv.df) is not set.")
        if _corpus is not None:
            _corpus.metadata["numeric_clusters"] = _numeric_clusters
            self._csv.corpus = _corpus
        return members

    def get_nnet_predictions(self, y: str, mcp=False, linkage_method: str | None = None, aggregation: str = "majority"):
        """
        Extended: Handles binary (BCELoss) and multi-class (CrossEntropyLoss).
        Returns list of predicted original class labels.

        Args:
            y (str): Target column name OR text metadata field name (when linkage_method is specified).
            mcp (bool): Whether to return MCP-formatted string.
            linkage_method (str, optional): Linkage method for text metadata outcomes ('id', 'embedding', 'temporal', 'keyword').
            aggregation (str): Aggregation strategy when multiple documents link to one row ('majority', 'mean', 'first', 'mode').
        """
        if ML_INSTALLED is False:
            logger.info(
                "ML dependencies are not installed. Please install them by ```pip install crisp-t[ml] to use ML features."
            )
            return None

        if self._csv is None:
            raise ValueError(
                "CSV data is not set. Please set self.csv before calling profile."
            )
        _corpus = self._csv.corpus

        X_np, Y_raw, X, Y = self._process_xy(y=y, linkage_method=linkage_method, aggregation=aggregation)

        unique_classes = np.unique(Y_raw)
        num_classes = unique_classes.size
        if num_classes < 2:
            raise ValueError(f"Need at least 2 classes; found {num_classes}.")

        vnum = X_np.shape[1]

        # Binary path
        if num_classes == 2:
            # Map to {0.0,1.0} for BCELoss if needed
            mapping_applied = False
            class_mapping = {}
            inverse_mapping = {}
            # Ensure deterministic order
            sorted_classes = sorted(unique_classes.tolist())
            if not (sorted_classes == [0, 1] or sorted_classes == [0.0, 1.0]):
                class_mapping = {sorted_classes[0]: 0.0, sorted_classes[1]: 1.0}
                inverse_mapping = {v: k for k, v in class_mapping.items()}
                Y_mapped = np.vectorize(class_mapping.get)(Y_raw).astype(np.float32)
                mapping_applied = True
            else:
                Y_mapped = Y_raw.astype(np.float32)

            model = NeuralNet(vnum)
            try:
                criterion = nn.BCELoss()  # type: ignore
                optimizer = optim.Adam(model.parameters(), lr=ml_config.NNET_LEARNING_RATE)  # type: ignore

                X_tensor = torch.from_numpy(X_np)  # type: ignore
                y_tensor = torch.from_numpy(Y_mapped.astype(np.float32)).view(-1, 1)  # type: ignore

                dataset = TensorDataset(X_tensor, y_tensor)  # type: ignore
                dataloader = DataLoader(dataset, batch_size=ml_config.NNET_BATCH_SIZE, shuffle=True)  # type: ignore
            except Exception as e:
                logger.exception(f"Error occurred while creating DataLoader: {e}")
                return None

            for _ in range(self._epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    if torch.isnan(loss):  # type: ignore
                        raise RuntimeError("NaN loss encountered.")
                    loss.backward()
                    optimizer.step()

            # Predictions
            bin_preds_internal = None
            if torch:
                with torch.no_grad():
                    probs = model(torch.from_numpy(X_np)).view(-1).cpu().numpy()
                bin_preds_internal = (probs >= 0.5).astype(int)

            if mapping_applied:
                preds = [inverse_mapping[float(p)] for p in bin_preds_internal]  # type: ignore
                y_eval = np.vectorize(class_mapping.get)(Y_raw).astype(int)
                preds_eval = bin_preds_internal
            else:
                preds = bin_preds_internal.tolist()  # type: ignore
                y_eval = Y_mapped.astype(int)
                preds_eval = bin_preds_internal

            accuracy = (preds_eval == y_eval).sum() / len(y_eval)
            print(
                f"\nPredicting {y} with {X.shape[1]} features for {self._epochs} epochs gave an accuracy (convergence): {accuracy*100:.2f}%\n"
            )
            if _corpus is not None:
                _corpus.metadata[ml_config.METADATA_KEY_NNET] = (
                    f"Predicting {y} with {X.shape[1]} features for {self._epochs} epochs gave an accuracy (convergence): {accuracy*100:.2f}%"
                )
            if mcp:
                return f"Predicting {y} with {X.shape[1]} features for {self._epochs} epochs gave an accuracy (convergence): {accuracy*100:.2f}%"
            return preds

        # Multi-class path
        # Map original classes to indices
        sorted_classes = sorted(unique_classes.tolist())
        class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        Y_idx = np.vectorize(class_to_idx.get)(Y_raw).astype(np.int64)

        model = MultiClassNet(vnum, num_classes)
        criterion = nn.CrossEntropyLoss()  # type: ignore
        optimizer = optim.Adam(model.parameters(), lr=ml_config.NNET_LEARNING_RATE)  # type: ignore

        X_tensor = torch.from_numpy(X_np)  # type: ignore
        y_tensor = torch.from_numpy(Y_idx)  # type: ignore

        dataset = TensorDataset(X_tensor, y_tensor)  # type: ignore
        dataloader = DataLoader(dataset, batch_size=ml_config.NNET_BATCH_SIZE, shuffle=True)  # type: ignore

        for _ in range(self._epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                if torch.isnan(loss):  # type: ignore
                    raise RuntimeError("NaN loss encountered.")
                loss.backward()
                optimizer.step()

        with torch.no_grad():  # type: ignore
            logits_full = model(torch.from_numpy(X_np))  # type: ignore
            pred_indices = torch.argmax(logits_full, dim=1).cpu().numpy()  # type: ignore

        preds = [idx_to_class[i] for i in pred_indices]
        accuracy = (pred_indices == Y_idx).sum() / len(Y_idx)
        print(
            f"\nPredicting {y} with {X.shape[1]} features for {self._epochs} gave an accuracy (convergence): {accuracy*100:.2f}%\n"
        )
        if _corpus is not None:
            _corpus.metadata[ml_config.METADATA_KEY_NNET] = (
                f"Predicting {y} with {X.shape[1]} features for {self._epochs} gave an accuracy (convergence): {accuracy*100:.2f}%"
            )
        if mcp:
            return f"Predicting {y} with {X.shape[1]} features for {self._epochs} gave an accuracy (convergence): {accuracy*100:.2f}%"
        return preds

    def _convert_to_binary(self, Y):
        unique_values = np.unique(Y)
        if len(unique_values) != 2:
            logger.warning(
                "Target variable has more than two unique values."
            )
            # convert unique_values[0] to 0, rest to 1
            mapping = {val: (0 if val == unique_values[0] else 1) for val in unique_values}
        else:
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
        Y_binary = np.vectorize(mapping.get)(Y)
        print(f"Converted target variable to binary using mapping: {mapping}")
        return Y_binary

    def svm_confusion_matrix(self, y: str, test_size=ml_config.CLASSIFIER_TEST_SIZE, random_state=ml_config.KMEANS_RANDOM_STATE, mcp=False, linkage_method: str | None = None, aggregation: str = "majority"):
        """Generate confusion matrix for SVM

        Args:
            y (str): Target column name OR text metadata field name (when linkage_method is specified).
            test_size (float): Proportion of dataset to include in test split.
            random_state (int): Random state for reproducibility.
            mcp (bool): Whether to return MCP-formatted string.
            linkage_method (str, optional): Linkage method for text metadata outcomes ('id', 'embedding', 'temporal', 'keyword').
            aggregation (str): Aggregation strategy when multiple documents link to one row ('majority', 'mean', 'first', 'mode').

        Returns:
            [list] -- [description]
        """
        X_np, Y_raw, X, Y = self._process_xy(y=y, linkage_method=linkage_method, aggregation=aggregation)
        Y = self._convert_to_binary(Y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        sc = StandardScaler()
        # Issue #22
        y_test = y_test.astype("int")
        y_train = y_train.astype("int")
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = SVC(kernel="linear", random_state=ml_config.KMEANS_RANDOM_STATE)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # Issue #22
        y_pred = y_pred.astype("int")
        _confusion_matrix = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for SVM predicting {y}:\n{_confusion_matrix}")
        # Output
        # [[2 0]
        #  [2 0]]
        if self._csv.corpus is not None:
            self._csv.corpus.metadata[ml_config.METADATA_KEY_SVM_CONFUSION] = (
                f"Confusion Matrix for SVM predicting {y}:\n{self.format_confusion_matrix_to_human_readable(_confusion_matrix)}"
            )

        if mcp:
            return f"Confusion Matrix for SVM predicting {y}:\n{self.format_confusion_matrix_to_human_readable(_confusion_matrix)}"

        return _confusion_matrix

    def format_confusion_matrix_to_human_readable(
        self, confusion_matrix: np.ndarray
    ) -> str:
        """Format the confusion matrix to a human-readable string.

        Args:
            confusion_matrix (np.ndarray): The confusion matrix to format.

        Returns:
            str: The formatted confusion matrix with true positive, false positive, true negative, and false negative counts.
        """
        tn, fp, fn, tp = confusion_matrix.ravel()
        return (
            f"True Positive: {tp}\n"
            f"False Positive: {fp}\n"
            f"True Negative: {tn}\n"
            f"False Negative: {fn}\n"
        )

    # https://stackoverflow.com/questions/45419203/python-numpy-extracting-a-row-from-an-array
    def knn_search(self, y: str, n=3, r=3, mcp=False, linkage_method: str | None = None, aggregation: str = "majority"):
        """
        Perform K-Nearest Neighbors search.

        Args:
            y (str): Target column name OR text metadata field name (when linkage_method is specified).
            n (int): Number of nearest neighbors to find.
            r (int): Record number to search from (1-based index).
            mcp (bool): Whether to return MCP-formatted string.
            linkage_method (str, optional): Linkage method for text metadata outcomes ('id', 'embedding', 'temporal', 'keyword').
            aggregation (str): Aggregation strategy when multiple documents link to one row ('majority', 'mean', 'first', 'mode').
        """
        X_np, Y_raw, X, Y = self._process_xy(y=y, linkage_method=linkage_method, aggregation=aggregation)
        kdt = KDTree(X_np, leaf_size=2, metric="euclidean")
        dist, ind = kdt.query(X_np[r - 1 : r, :], k=n)
        # Display results as human readable (1-based)
        ind = (ind + 1).tolist()  # Convert to 1-based index
        dist = dist.tolist()
        print(
            f"\nKNN search for {y} (n={n}, record no: {r}): {ind} with distances {dist}\n"
        )
        if self._csv.corpus is not None:
            self._csv.corpus.metadata["knn_search"] = (
                f"KNN search for {y} (n={n}, record no: {r}): {ind} with distances {dist}"
            )
        if mcp:
            return f"KNN search for {y} (n={n}, record no: {r}): {ind} with distances {dist}"
        return dist, ind

    def _extract_outcome_from_text_metadata(
        self,
        metadata_field: str,
        linkage_method: str,
        aggregation: str = "majority"
    ) -> tuple[pd.Series, list[int]]:
        """
        Extract outcome values from text document metadata using specified linkage method.

        Args:
            metadata_field (str): Name of the metadata field in documents containing outcome values
            linkage_method (str): Linkage method to use ('id', 'embedding', 'temporal', 'keyword')
            aggregation (str): Strategy for aggregating multiple documents linked to same row
                              - 'majority': Majority vote for classification (default)
                              - 'mean': Mean value for regression
                              - 'first': Use first document's value
                              - 'mode': Mode (most common) value

        Returns:
            Tuple[pd.Series, List[int]]: (outcome_series, valid_indices)
                - outcome_series: Series with outcome values aligned to DataFrame rows
                - valid_indices: List of DataFrame indices that have linked documents
        """
        if not self._csv.corpus or not self._csv.corpus.documents:
            raise ValueError("Corpus documents not available. Cannot extract text metadata outcomes.")

        if self._csv.df is None:
            raise ValueError("DataFrame not available.")

        # Build mapping from DataFrame index to document metadata values
        df_index_to_values: dict[int, list] = {}

        if linkage_method == "id":
            # ID linkage: Match document.id with df['id'] column
            if "id" not in self._csv.df.columns:
                raise ValueError("ID linkage requires 'id' column in DataFrame")

            # Create a mapping from id to df index
            id_to_df_index = {
                str(row_id): idx
                for idx, row_id in self._csv.df['id'].items()
            }

            for doc in self._csv.corpus.documents:
                if metadata_field in doc.metadata:
                    doc_id = str(doc.id)
                    if doc_id in id_to_df_index:
                        df_idx = id_to_df_index[doc_id]
                        if df_idx not in df_index_to_values:
                            df_index_to_values[df_idx] = []
                        df_index_to_values[df_idx].append(doc.metadata[metadata_field])

        elif linkage_method in ["embedding", "temporal"]:
            # Embedding or temporal linkage: Use links stored in document metadata
            link_key = f"{linkage_method}_links"

            for doc in self._csv.corpus.documents:
                if metadata_field in doc.metadata and link_key in doc.metadata:
                    links = doc.metadata[link_key]
                    if isinstance(links, list):
                        for link in links:
                            if isinstance(link, dict) and "df_index" in link:
                                df_idx = link["df_index"]
                                if df_idx not in df_index_to_values:
                                    df_index_to_values[df_idx] = []
                                df_index_to_values[df_idx].append(doc.metadata[metadata_field])

        elif linkage_method == "keyword":
            # Keyword linkage: Use keyword_links stored in document metadata
            for doc in self._csv.corpus.documents:
                if metadata_field in doc.metadata and "keyword_links" in doc.metadata:
                    links = doc.metadata["keyword_links"]
                    if isinstance(links, list):
                        for link in links:
                            if isinstance(link, dict) and "df_index" in link:
                                df_idx = link["df_index"]
                                if df_idx not in df_index_to_values:
                                    df_index_to_values[df_idx] = []
                                df_index_to_values[df_idx].append(doc.metadata[metadata_field])

        else:
            raise ValueError(
                f"Unsupported linkage method: {linkage_method}. "
                f"Supported methods: 'id', 'embedding', 'temporal', 'keyword'"
            )

        if not df_index_to_values:
            raise ValueError(
                f"No documents with '{metadata_field}' metadata field are linked to DataFrame rows "
                f"using '{linkage_method}' linkage method."
            )

        # Aggregate values for each DataFrame row
        aggregated_values = {}
        for df_idx, values in df_index_to_values.items():
            if aggregation == "majority":
                # Majority vote (for classification)
                counter = Counter(values)
                aggregated_values[df_idx] = counter.most_common(1)[0][0]
            elif aggregation == "mean":
                # Mean value (for regression)
                try:
                    numeric_values = [float(v) for v in values]
                    aggregated_values[df_idx] = np.mean(numeric_values)
                except (ValueError, TypeError):
                    # Fall back to majority if not numeric
                    logger.warning(
                        f"Non-numeric values found for df_index {df_idx}, using majority vote instead"
                    )
                    counter = Counter(values)
                    aggregated_values[df_idx] = counter.most_common(1)[0][0]
            elif aggregation == "first":
                # Use first document's value
                aggregated_values[df_idx] = values[0]
            elif aggregation == "mode":
                # Mode (most common value)
                counter = Counter(values)
                aggregated_values[df_idx] = counter.most_common(1)[0][0]
            else:
                raise ValueError(
                    f"Unsupported aggregation method: {aggregation}. "
                    f"Supported methods: 'majority', 'mean', 'first', 'mode'"
                )

        # Create Series aligned with DataFrame
        valid_indices = sorted(aggregated_values.keys())
        outcome_series = pd.Series(
            [aggregated_values[idx] for idx in valid_indices],
            index=valid_indices,
            name=metadata_field
        )

        logger.info(
            f"Extracted outcome from text metadata field '{metadata_field}' "
            f"using '{linkage_method}' linkage and '{aggregation}' aggregation. "
            f"Linked {len(valid_indices)} out of {len(self._csv.df)} DataFrame rows."
        )

        return outcome_series, valid_indices

    def _process_xy(self, y: str, oversample=False, one_hot_encode_all=False, linkage_method: str | None = None, aggregation: str = "majority"):
        """
        Process features and outcome data for ML models.

        Args:
            y (str): Either a DataFrame column name OR a text metadata field name (when linkage_method is specified)
            oversample (bool): Whether to oversample minority classes
            one_hot_encode_all (bool): Whether to one-hot encode all columns
            linkage_method (str, optional): Linkage method for text metadata outcomes ('id', 'embedding', 'temporal', 'keyword')
            aggregation (str): Aggregation strategy when multiple documents link to one row ('majority', 'mean', 'first', 'mode')

        Returns:
            Tuple: (X_np, Y_raw, X, Y) - Feature matrix, outcome array, original DataFrames
        """
        # Check if y is a text metadata field (requires linkage_method)
        if linkage_method:
            # Extract outcome from text metadata using specified linkage
            outcome_series, valid_indices = self._extract_outcome_from_text_metadata(
                metadata_field=y,
                linkage_method=linkage_method,
                aggregation=aggregation
            )

            # Filter DataFrame to only include rows with linked documents
            filtered_df = self._csv.df.loc[valid_indices].copy()

            # Add the outcome column to the filtered DataFrame
            filtered_df[f"_text_outcome_{y}"] = outcome_series

            # Temporarily swap the DataFrame
            original_df = self._csv.df
            self._csv.df = filtered_df

            try:
                # Prepare data using the temporary outcome column
                X, Y = self._csv.prepare_data(
                    y=f"_text_outcome_{y}",
                    oversample=oversample,
                    one_hot_encode_all=one_hot_encode_all
                )
            finally:
                # Restore original DataFrame and remove temporary column
                filtered_df.drop(columns=[f"_text_outcome_{y}"], inplace=True)
                self._csv.df = original_df

            # Rename Y back to the original metadata field name
            if hasattr(Y, 'name'):
                Y.name = y
        else:
            # Standard path: y is a DataFrame column
            X, Y = self._csv.prepare_data(
                y=y, oversample=oversample, one_hot_encode_all=one_hot_encode_all
            )

        if X is None or Y is None:
            raise ValueError("prepare_data returned None for X or Y.")

        # To numpy float32
        X_np = (
            X.to_numpy(dtype=np.float32)
            if hasattr(X, "to_numpy")
            else np.asarray(X, dtype=np.float32)
        )
        Y_raw = Y.to_numpy() if hasattr(Y, "to_numpy") else np.asarray(Y)

        # Handle NaNs
        if np.isnan(X_np).any():
            raise ValueError("NaN detected in feature matrix.")
        if np.isnan(Y_raw.astype(float, copy=False)).any():
            raise ValueError("NaN detected in target vector.")

        return X_np, Y_raw, X, Y

    def get_decision_tree_classes(
        self, y: str, top_n=5, test_size=0.5, random_state=1, mcp=False, linkage_method: str | None = None, aggregation: str = "majority"
    ):
        """
        Train a Decision Tree classifier and return feature importances.

        Args:
            y (str): Target column name OR text metadata field name (when linkage_method is specified).
            top_n (int): Number of top features to display.
            test_size (float): Proportion of dataset to include in test split.
            random_state (int): Random state for reproducibility.
            mcp (bool): Whether to return MCP-formatted string.
            linkage_method (str, optional): Linkage method for text metadata outcomes ('id', 'embedding', 'temporal', 'keyword').
            aggregation (str): Aggregation strategy when multiple documents link to one row ('majority', 'mean', 'first', 'mode').

        Returns:
            dict or str: Feature importances and accuracy metrics.
        """
        X_np, Y_raw, X, Y = self._process_xy(y=y, linkage_method=linkage_method, aggregation=aggregation)
        Y_raw = self._convert_to_binary(Y_raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, Y_raw, test_size=test_size, random_state=random_state
        )

        # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        # print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Train a RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=ml_config.CLASSIFIER_RANDOM_STATE)
        clf.fit(X_train, y_train)

        # Compute permutation importance
        results = permutation_importance(
            clf, X_test, y_test, n_repeats=10, random_state=ml_config.CLASSIFIER_RANDOM_STATE
        )

        # classifier = DecisionTreeClassifier(random_state=random_state) # type: ignore
        # classifier.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        _confusion_matrix = confusion_matrix(y_test, y_pred)
        print(
            f"Confusion Matrix for Decision Tree predicting {y}:\n{_confusion_matrix}"
        )
        # Output
        # [[2 0]
        #  [2 0]]

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy}\n")

        # Retrieve feature importance scores
        importance = results.importances_mean

        # Get indices of top N important features
        top_n_indices = np.argsort(importance)[-top_n:][::-1]

        # Display feature importance
        print(f"==== Top {top_n} important features ====\n")
        _importance = ""
        for i, v in enumerate(top_n_indices):
            print(f"Feature: {X.columns[v]}, Score: {importance[v]:.5f}")
            _importance += f"Feature: {X.columns[v]}, Score: {importance[v]:.5f}\n"

        if self._csv.corpus is not None:
            self._csv.corpus.metadata["decision_tree_accuracy"] = (
                f"Decision Tree accuracy for predicting {y}: {accuracy*100:.2f}%"
            )
            self._csv.corpus.metadata["decision_tree_confusion_matrix"] = (
                f"Confusion Matrix for Decision Tree predicting {y}:\n{self.format_confusion_matrix_to_human_readable(_confusion_matrix)}"
            )
            self._csv.corpus.metadata["decision_tree_feature_importance"] = _importance
        if mcp:
            return f"""
            Confusion Matrix for Decision Tree predicting {y}:\n{self.format_confusion_matrix_to_human_readable(_confusion_matrix)}\nTop {top_n} important features:\n{_importance}
            Accuracy: {accuracy*100:.2f}%
            """
        return _confusion_matrix, importance

    def get_xgb_classes(
        self, y: str, oversample=False, test_size=ml_config.CLASSIFIER_TEST_SIZE, random_state=ml_config.KMEANS_RANDOM_STATE, mcp=False, linkage_method: str | None = None, aggregation: str = "majority"
    ):
        """
        Train an XGBoost classifier and return feature importances.

        Args:
            y (str): Target column name OR text metadata field name (when linkage_method is specified).
            oversample (bool): Whether to oversample minority classes.
            test_size (float): Proportion of dataset to include in test split.
            random_state (int): Random state for reproducibility.
            mcp (bool): Whether to return MCP-formatted string.
            linkage_method (str, optional): Linkage method for text metadata outcomes ('id', 'embedding', 'temporal', 'keyword').
            aggregation (str): Aggregation strategy when multiple documents link to one row ('majority', 'mean', 'first', 'mode').

        Returns:
            dict or str: Feature importances and accuracy metrics.
        """
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Please install it via `pip install crisp-t[xg]`."
            ) from None
        X_np, Y_raw, X, Y = self._process_xy(y=y, linkage_method=linkage_method, aggregation=aggregation)
        if ML_INSTALLED:
            # ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1], got [1 2]
            # convert y to binary
            Y_binary = (Y_raw == 1).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, Y_binary, test_size=test_size, random_state=random_state
            )
            classifier = XGBClassifier(use_label_encoder=False, eval_metric="logloss")  # type: ignore
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            _confusion_matrix = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix for XGBoost predicting {y}:\n{_confusion_matrix}")
            # Output
            # [[2 0]
            #  [2 0]]
            if self._csv.corpus is not None:
                self._csv.corpus.metadata["xgb_confusion_matrix"] = (
                    f"Confusion Matrix for XGBoost predicting {y}:\n{_confusion_matrix}"
                )
            if mcp:
                return f"""
                Confusion Matrix for XGBoost predicting {y}:\n{self.format_confusion_matrix_to_human_readable(_confusion_matrix)}
                """
            return _confusion_matrix
        else:
            raise ImportError("ML dependencies are not installed.")

    def get_apriori(
        self, y: str, min_support=0.9, use_colnames=True, min_threshold=0.5, mcp=False
    ):
        if ML_INSTALLED:
            X_np, Y_raw, X, Y = self._process_xy(y=y, one_hot_encode_all=True)
            frequent_itemsets = apriori(X, min_support=min_support, use_colnames=use_colnames)  # type: ignore
            # rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold) # type: ignore
            human_readable = tabulate(
                frequent_itemsets.head(10), headers="keys", tablefmt="pretty"  # type: ignore
            )
            if self._csv.corpus is not None:
                self._csv.corpus.metadata["apriori_frequent_itemsets"] = human_readable
            if mcp:
                return f"Frequent itemsets (top 10):\n{human_readable}"
            return frequent_itemsets  # , rules
        else:
            raise ImportError("ML dependencies are not installed.")

    def get_pca(self, y: str, n: int = 3, mcp=False, linkage_method: str | None = None, aggregation: str = "majority"):
        """
        Perform a manual PCA (no sklearn PCA) on the feature matrix for target y.

        Args:
            y (str): Target column name (used only for data preparation) OR text metadata field name (when linkage_method is specified).
            n (int): Number of principal components to keep.
            mcp (bool): Whether to return MCP-formatted string.
            linkage_method (str, optional): Linkage method for text metadata outcomes ('id', 'embedding', 'temporal', 'keyword').
            aggregation (str): Aggregation strategy when multiple documents link to one row ('majority', 'mean', 'first', 'mode').

        Returns:
            dict: {
                'covariance_matrix': cov_mat,
                'eigenvalues': eig_vals_sorted,
                'eigenvectors': eig_vecs_sorted,
                'explained_variance_ratio': var_exp,
                'cumulative_explained_variance_ratio': cum_var_exp,
                'projection_matrix': matrix_w,
                'transformed': X_pca
            }
        """
        X_np, Y_raw, X, Y = self._process_xy(y=y, linkage_method=linkage_method, aggregation=aggregation)
        X_std = StandardScaler().fit_transform(X_np)

        cov_mat = np.cov(X_std.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)  # symmetric matrix -> eigh

        # Sort eigenvalues (and vectors) descending
        idx = np.argsort(eig_vals)[::-1]
        eig_vals_sorted = eig_vals[idx]
        eig_vecs_sorted = eig_vecs[:, idx]

        factors = X_std.shape[1]
        n = max(1, min(n, factors))

        # Explained variance ratios
        tot = eig_vals_sorted.sum()
        var_exp = (eig_vals_sorted / tot) * 100.0
        cum_var_exp = np.cumsum(var_exp)

        # Projection matrix (first n eigenvectors)
        matrix_w = eig_vecs_sorted[:, :n]

        # Project data
        X_pca = X_std @ matrix_w

        # Optional prints (retain original behavior)
        print("Covariance matrix:\n", cov_mat)
        print("Eigenvalues (desc):\n", eig_vals_sorted)
        print("Explained variance (%):\n", var_exp[:n])
        print("Cumulative explained variance (%):\n", cum_var_exp[:n])
        print("Projection matrix (W):\n", matrix_w)
        print("Transformed (first 5 rows):\n", X_pca[:5])

        result = {
            "covariance_matrix": cov_mat,
            "eigenvalues": eig_vals_sorted,
            "eigenvectors": eig_vecs_sorted,
            "explained_variance_ratio": var_exp,
            "cumulative_explained_variance_ratio": cum_var_exp,
            "projection_matrix": matrix_w,
            "transformed": X_pca,
        }

        if self._csv.corpus is not None:
            self._csv.corpus.metadata[ml_config.METADATA_KEY_PCA] = (
                f"PCA kept {n} components explaining "
                f"{cum_var_exp[n-1]:.2f}% variance."
            )
        if mcp:
            return (
                f"PCA kept {n} components explaining {cum_var_exp[n-1]:.2f}% variance."
            )
        return result

    def get_regression(self, y: str, mcp=False, linkage_method: str | None = None, aggregation: str = "mean"):
        """
        Perform linear or logistic regression based on the outcome variable type.

        If the outcome is binary, fit a logistic regression model.
        Otherwise, fit a linear regression model.

        Args:
            y (str): Target column name for the regression OR text metadata field name (when linkage_method is specified).
            mcp (bool): Whether to return MCP-formatted string.
            linkage_method (str, optional): Linkage method for text metadata outcomes ('id', 'embedding', 'temporal', 'keyword').
            aggregation (str): Aggregation strategy when multiple documents link to one row ('majority', 'mean', 'first', 'mode').
                             Default is 'mean' for regression tasks.

        Returns:
            dict: Regression results including coefficients, intercept, and metrics.
        """
        if ML_INSTALLED is False:
            logger.info(
                "ML dependencies are not installed. Please install them by ```pip install crisp-t[ml] to use ML features."
            )
            return None

        if self._csv is None:
            raise ValueError(
                "CSV data is not set. Please set self.csv before calling get_regression."
            )

        X_np, Y_raw, X, Y = self._process_xy(y=y, linkage_method=linkage_method, aggregation=aggregation)

        # Check if outcome is binary (logistic) or continuous (linear)
        unique_values = np.unique(Y_raw)
        num_unique = len(unique_values)

        # Determine if binary classification or regression
        is_binary = num_unique == 2

        if is_binary:
            # Logistic Regression
            print(f"\n=== Logistic Regression for {y} ===")
            print(f"Binary outcome detected with values: {unique_values}")

            model = LogisticRegression(max_iter=1000, random_state=ml_config.CLASSIFIER_RANDOM_STATE)
            model.fit(X_np, Y_raw)

            # Predictions
            y_pred = model.predict(X_np)

            # Accuracy
            accuracy = accuracy_score(Y_raw, y_pred)
            print(f"\nAccuracy: {accuracy*100:.2f}%")

            # Coefficients and Intercept
            print("\nCoefficients:")
            for i, coef in enumerate(model.coef_[0]):
                feature_name = X.columns[i] if hasattr(X, "columns") else f"Feature_{i}"
                print(f"  {feature_name}: {coef:.5f}")

            print(f"\nIntercept: {model.intercept_[0]:.5f}")

            coef_str = "\n".join(
                [
                    f"  {X.columns[i] if hasattr(X, 'columns') else f'Feature_{i}'}: {coef:.5f}"
                    for i, coef in enumerate(model.coef_[0])
                ]
            )

            # Store in metadata
            if self._csv.corpus is not None:
                self._csv.corpus.metadata["logistic_regression_accuracy"] = (
                    f"Logistic Regression accuracy for predicting {y}: {accuracy*100:.2f}%"
                )
                self._csv.corpus.metadata["logistic_regression_coefficients"] = (
                    f"Coefficients:\n{coef_str}"
                )
                self._csv.corpus.metadata["logistic_regression_intercept"] = (
                    f"Intercept: {model.intercept_[0]:.5f}"
                )

            if mcp:
                return f"""
                Logistic Regression accuracy for predicting {y}: {accuracy*100:.2f}%
                Coefficients:
                {coef_str}
                Intercept: {model.intercept_[0]:.5f}
                """
            return {
                "model_type": "logistic",
                "accuracy": accuracy,
                "coefficients": model.coef_[0],
                "intercept": model.intercept_[0],
                "feature_names": X.columns.tolist() if hasattr(X, "columns") else None,
            }
        else:
            # Linear Regression
            print(f"\n=== Linear Regression for {y} ===")
            print(f"Continuous outcome detected with {num_unique} unique values")

            model = LinearRegression()
            model.fit(X_np, Y_raw)

            # Predictions
            y_pred = model.predict(X_np)

            # Metrics
            mse = mean_squared_error(Y_raw, y_pred)
            r2 = r2_score(Y_raw, y_pred)
            print(f"\nMean Squared Error (MSE): {mse:.5f}")
            print(f"R² Score: {r2:.5f}")

            # Coefficients and Intercept
            print("\nCoefficients:")
            for i, coef in enumerate(model.coef_):
                feature_name = X.columns[i] if hasattr(X, "columns") else f"Feature_{i}"
                print(f"  {feature_name}: {coef:.5f}")

            print(f"\nIntercept: {model.intercept_:.5f}")

            coef_str = "\n".join(
                [
                    f"  {X.columns[i] if hasattr(X, 'columns') else f'Feature_{i}'}: {coef:.5f}"
                    for i, coef in enumerate(model.coef_)
                ]
            )

            # Store in metadata
            if self._csv.corpus is not None:
                self._csv.corpus.metadata["linear_regression_mse"] = (
                    f"Linear Regression MSE for predicting {y}: {mse:.5f}"
                )
                self._csv.corpus.metadata["linear_regression_r2"] = (
                    f"Linear Regression R² for predicting {y}: {r2:.5f}"
                )
                self._csv.corpus.metadata["linear_regression_coefficients"] = (
                    f"Coefficients:\n{coef_str}"
                )
                self._csv.corpus.metadata["linear_regression_intercept"] = (
                    f"Intercept: {model.intercept_:.5f}"
                )

            if mcp:
                return f"""
                Linear Regression MSE for predicting {y}: {mse:.5f}
                R²: {r2:.5f}
                Feature Names and Coefficients:
                {coef_str}
                Intercept: {model.intercept_:.5f}
                """
            return {
                "model_type": "linear",
                "mse": mse,
                "r2": r2,
                "coefficients": model.coef_,
                "intercept": model.intercept_,
                "feature_names": X.columns.tolist() if hasattr(X, "columns") else None,
            }

    def get_lstm_predictions(self, y: str, mcp=False):
        """
        Train an LSTM model on text data to predict an outcome variable.
        This tests if the texts converge towards predicting the outcome.

        Args:
            y (str): Name of the outcome variable in the DataFrame
            mcp (bool): If True, return a string format suitable for MCP

        Returns:
            Evaluation metrics as string (if mcp=True) or dict
        """
        if ML_INSTALLED is False:
            logger.error(
                "ML dependencies are not installed. Please install them by ```pip install crisp-t[ml] to use ML features."
            )
            if mcp:
                return "ML dependencies are not installed. Please install with: pip install crisp-t[ml]"
            return None

        if self._csv is None:
            logger.error("CSV data is not set.")
            if mcp:
                return "CSV data is not set. Cannot perform LSTM prediction."
            return None

        _corpus = self._csv.corpus
        if _corpus is None:
            logger.error("Corpus is not available.")
            if mcp:
                return "Corpus is not available. Cannot perform LSTM prediction."
            return None

        # Check if id_column exists
        id_column = "id"
        if not hasattr(self._csv, "df") or self._csv.df is None:
            logger.error("DataFrame is not available in CSV.")
            if mcp:
                return "This tool can be used only if texts and outcome variables align. DataFrame is missing."
            return None

        if id_column not in self._csv.df.columns:
            logger.error(
                f"The id_column '{id_column}' does not exist in the DataFrame."
            )
            if mcp:
                return f"This tool can be used only if texts and outcome variables align. The '{id_column}' column is missing from the DataFrame."
            return None

        # Check if outcome variable exists
        if y not in self._csv.df.columns:
            logger.error(f"The outcome variable '{y}' does not exist in the DataFrame.")
            if mcp:
                return f"The outcome variable '{y}' does not exist in the DataFrame."
            return None

        # Process documents and align with outcome variable
        try:
            # Build vocabulary from all documents
            from collections import Counter

            word_counts = Counter()
            tokenized_docs = []

            for doc in tqdm(_corpus.documents, desc="Tokenizing documents", disable=len(_corpus.documents) < 10):
                # Simple tokenization - split on whitespace and lowercase
                tokens = doc.text.lower().split()
                tokenized_docs.append(tokens)
                word_counts.update(tokens)

            # Create vocabulary with most common words (limit to 10000)
            vocab_size = min(ml_config.LSTM_VOCAB_SIZE, len(word_counts)) + 1  # +1 for padding
            most_common = word_counts.most_common(vocab_size - 1)
            word_to_idx = {
                word: idx + 1 for idx, (word, _) in enumerate(most_common)
            }  # 0 reserved for padding

            # Convert documents to sequences of indices
            max_length = ml_config.LSTM_MAX_LENGTH  # Maximum sequence length
            sequences = []
            doc_ids = []

            for doc, tokens in tqdm(zip(_corpus.documents, tokenized_docs), total=len(_corpus.documents), desc="Converting to sequences", disable=len(_corpus.documents) < 10):
                # Convert tokens to indices
                seq = [word_to_idx.get(token, 0) for token in tokens]
                # Pad or truncate to max_length
                if len(seq) > max_length:
                    seq = seq[:max_length]
                else:
                    seq = seq + [0] * (max_length - len(seq))
                sequences.append(seq)
                doc_ids.append(doc.id)

            # Align with outcome variable using id column
            df = self._csv.df.set_index(id_column)

            aligned_sequences = []
            aligned_outcomes = []

            df_index_str = list(str(idx) for idx in df.index)
            for doc_id, seq in zip(doc_ids, sequences):
                if doc_id in df_index_str:
                    aligned_sequences.append(seq)
                    # Select y from df where id_column == doc_id, using string comparison
                    matched_row = df.loc[
                        [idx for idx in df.index if str(idx) == str(doc_id)]
                    ]
                    if not matched_row.empty:
                        aligned_outcomes.append(matched_row.iloc[0][y])

            if len(aligned_sequences) == 0:
                logger.error("No documents could be aligned with the outcome variable.")
                if mcp:
                    return "This tool can be used only if texts and outcome variables align. No matching IDs found."
                return None

            # Convert to tensors
            X_tensor = torch.LongTensor(aligned_sequences)  # type: ignore
            y_array = np.array(aligned_outcomes)

            # Handle binary classification
            unique_values = np.unique(y_array)
            num_classes = len(unique_values)

            if num_classes < 2:
                logger.error(
                    f"Need at least 2 classes for classification, found {num_classes}"
                )
                if mcp:
                    return f"Need at least 2 classes for classification, found {num_classes}"
                return None

            # Map to 0/1 for binary classification
            if num_classes == 2:
                class_mapping = {unique_values[0]: 0.0, unique_values[1]: 1.0}
                y_mapped = np.array(
                    [class_mapping[val] for val in y_array], dtype=np.float32
                )
            else:
                # Multi-class not supported in this simple LSTM implementation
                logger.error(
                    "Multi-class classification is not supported for LSTM. Please use binary outcome."
                )
                if mcp:
                    return "Multi-class classification is not supported for LSTM. Please use binary outcome."
                return None

            y_tensor = torch.FloatTensor(y_mapped).view(-1, 1)  # type: ignore

            # Split into train/test
            from sklearn.model_selection import train_test_split

            indices = list(range(len(X_tensor)))
            train_idx, test_idx = train_test_split(
                indices, test_size=ml_config.CLASSIFIER_TEST_SIZE, random_state=ml_config.CLASSIFIER_RANDOM_STATE
            )

            X_train = X_tensor[train_idx]
            y_train = y_tensor[train_idx]
            X_test = X_tensor[test_idx]
            y_test = y_tensor[test_idx]

            # Create model
            model = SimpleLSTM(vocab_size=vocab_size)  # type: ignore
            criterion = nn.BCELoss()  # type: ignore
            optimizer = optim.Adam(model.parameters(), lr=ml_config.LSTM_LEARNING_RATE)  # type: ignore

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)  # type: ignore
            train_loader = DataLoader(train_dataset, batch_size=ml_config.LSTM_BATCH_SIZE, shuffle=True)  # type: ignore

            # Training
            epochs = max(self._epochs, ml_config.LSTM_EPOCHS)  # Use at least configured epochs for LSTM
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # Evaluation
            model.eval()
            with torch.no_grad():  # type: ignore
                train_preds = model(X_train)
                test_preds = model(X_test)

                train_preds_binary = (train_preds >= 0.5).float()
                test_preds_binary = (test_preds >= 0.5).float()

                train_accuracy = (train_preds_binary == y_train).float().mean().item()
                test_accuracy = (test_preds_binary == y_test).float().mean().item()

            # Calculate additional metrics for test set
            y_test_np = y_test.cpu().numpy().flatten()
            test_preds_np = test_preds_binary.cpu().numpy().flatten()

            # Confusion matrix elements
            tp = ((test_preds_np == 1) & (y_test_np == 1)).sum()
            tn = ((test_preds_np == 0) & (y_test_np == 0)).sum()
            fp = ((test_preds_np == 1) & (y_test_np == 0)).sum()
            fn = ((test_preds_np == 0) & (y_test_np == 1)).sum()

            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            result_msg = (
                f"LSTM Model Evaluation for predicting '{y}':\n"
                f"  Vocabulary size: {vocab_size}\n"
                f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}\n"
                f"  Epochs: {epochs}\n"
                f"  Train accuracy: {train_accuracy*100:.2f}%\n"
                f"  Test accuracy (convergence): {test_accuracy*100:.2f}%\n"
                f"  True Positive: {tp}, False Positive: {fp}, True Negative: {tn}, False Negative: {fn}\n"
                f"  Precision: {precision:.3f}\n"
                f"  Recall: {recall:.3f}\n"
                f"  F1-Score: {f1:.3f}\n"
            )

            print(f"\n{result_msg}")

            # Store in corpus metadata
            if _corpus is not None:
                _corpus.metadata[ml_config.METADATA_KEY_LSTM] = result_msg

            if mcp:
                return result_msg

            return {
                "vocab_size": vocab_size,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "epochs": epochs,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        except Exception as e:
            logger.exception(f"Error in LSTM prediction: {e}")
            if mcp:
                return f"Error in LSTM prediction: {e}"
            return None
