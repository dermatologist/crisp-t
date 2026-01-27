"""Configuration constants for machine learning models.

This module centralizes hyperparameters, constants, and metadata keys
used across CRISP-T machine learning functions to reduce hardcoded values
and improve maintainability.
"""

# Neural Network Hyperparameters
NNET_BATCH_SIZE = 32
NNET_EPOCHS = 3
NNET_LEARNING_RATE = 0.001
NNET_HIDDEN_SIZE = 128

# LSTM Hyperparameters
LSTM_VOCAB_SIZE = 10000
LSTM_MAX_LENGTH = 100
LSTM_EMBEDDING_DIM = 128
LSTM_HIDDEN_DIM = 256
LSTM_EPOCHS = 3
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001

# Clustering
KMEANS_DEFAULT_CLUSTERS = 3
KMEANS_RANDOM_STATE = 0

# Classification
CLASSIFIER_TEST_SIZE = 0.2
CLASSIFIER_RANDOM_STATE = 42

# Association Rules
APRIORI_MIN_SUPPORT_DEFAULT = 0.1
APRIORI_MIN_THRESHOLD_DEFAULT = 0.5

# Metadata Keys (used when storing results in corpus.metadata)
METADATA_KEY_KMEANS = "kmeans"
METADATA_KEY_SVM_CONFUSION = "svm_confusion_matrix"
METADATA_KEY_NNET = "nnet_predictions"
METADATA_KEY_KNN = "knn_results"
METADATA_KEY_DECISION_TREE = "decision_tree"
METADATA_KEY_XGB = "xgboost"
METADATA_KEY_PCA = "pca"
METADATA_KEY_REGRESSION = "regression"
METADATA_KEY_APRIORI = "apriori"
METADATA_KEY_LSTM = "lstm_predictions"

# Aggregation Methods (for text metadata outcomes)
AGGREGATION_MODE = "mode"  # For classification
AGGREGATION_MEAN = "mean"  # For regression
AGGREGATION_MAJORITY = "majority"  # Alias for mode
AGGREGATION_MEDIAN = "median"  # Alternative for regression

# Linkage Methods (for text metadata outcomes)
LINKAGE_METHOD_ID = "id"
LINKAGE_METHOD_KEYWORD = "keyword"
LINKAGE_METHOD_TIME = "time"
LINKAGE_METHOD_EMBEDDING = "embedding"
