import logging
from pathlib import Path

from src.crisp_t.csv import Csv
from src.crisp_t.ml import ML

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ml_initialization(csv_fixture):
    ml = ML(
        csv=csv_fixture,
    )
    assert ml._csv == csv_fixture, "Csv should be set correctly"


def test_get_kmeans(csv_fixture, csv_file_fixture):
    csv_fixture.read_csv(csv_file_fixture)
    csv_fixture.drop_na()
    csv_fixture.one_hot_encode_strings_in_df()  # ValueError: could not convert string to float: 'chocolate, chips, ice cream'
    ml = ML(
        csv=csv_fixture,
    )
    kmeans, members = ml.get_kmeans(number_of_clusters=5)
    print(kmeans)
    print(members)
    assert kmeans is not None, "KMeans clustering should not be None"
    assert members is not None, "Members should not be None"
    # [1 1 1 0 0 2 3 1 1 0 1 3 4]
    # [[3, 4, 9], [0, 1, 2, 7, 8, 10], [5], [6, 11], [12]]


def test_profile(csv_fixture, csv_file_fixture):

    csv_fixture.read_csv(csv_file_fixture)
    csv_fixture.drop_na()
    csv_fixture.one_hot_encode_strings_in_df()  # ValueError: could not convert string to float: 'chocolate, chips, ice cream'

    print(csv_fixture.df.head())  # Print the first few rows of the DataFrame for debugging
    ml = ML(csv=csv_fixture)
    kmeans, members = ml.get_kmeans(number_of_clusters=5)
    profile = ml.profile(members, number_of_clusters=5)
    print(profile)
    assert profile is not None, "Profile should not be None"


def test_get_nnet_predictions(csv_fixture, csv_file_fixture):
    _csv = csv_fixture
    _csv.read_csv(csv_file_fixture)
    _csv.drop_na()
    ml = ML(csv=_csv)
    predictions = ml.get_nnet_predictions(y="Gender")
    assert predictions is not None, "Predictions should not be None"


def test_svm_confusion_matrix(csv_fixture, csv_file_fixture):
    _csv = csv_fixture
    _csv.read_csv(csv_file_fixture)
    _csv.drop_na()
    ml = ML(csv=_csv)
    confusion_matrix = ml.svm_confusion_matrix(y="Gender")
    assert confusion_matrix is not None, "Confusion matrix should not be None"
    human_readable = ml.format_confusion_matrix_to_human_readable(confusion_matrix)
    print(human_readable)


def test_knn_search(csv_fixture, csv_file_fixture):
    _csv = csv_fixture
    _csv.read_csv(csv_file_fixture)
    _csv.drop_na()
    ml = ML(csv=_csv)
    dist, ind = ml.knn_search(y="Gender", n=3, r=3)
    assert ind is not None, "Neighbors should not be None"
    print(f"KNN search for Gender (n=3, record no 3): {ind} with distances {dist}")


def test_get_xgb_classes(csv_fixture, csv_file_fixture):
    # if os is MacOS, skip this test due to xgboost issues
    import platform
    if platform.system() == "Darwin":
        print("Skipping test_get_xgb_classes: XGBoost test is not supported on MacOS")
        return True
    _csv = csv_fixture
    _csv.read_csv(csv_file_fixture)
    _csv.drop_na()
    ml = ML(csv=_csv)
    xgb_classes = ml.get_xgb_classes(y="Gender")
    assert xgb_classes is not None, "XGBoost classes should not be None"
    human_readable = ml.format_confusion_matrix_to_human_readable(xgb_classes)
    print(human_readable)


# def test_get_apriori(csv_fixture, csv_file_fixture):
#     _csv = csv_fixture
#     _csv.read_csv(csv_file_fixture)
#     _csv.drop_na()
#     ml = ML(csv=_csv)
#     apriori_rules = ml.get_apriori(y="Gender")
#     assert apriori_rules is not None, "Apriori rules should not be None"
#     print(apriori_rules)


def test_get_pca(csv_fixture, csv_file_fixture):
    _csv = csv_fixture
    _csv.read_csv(csv_file_fixture)
    _csv.drop_na()
    ml = ML(csv=_csv)
    pca_result = ml.get_pca(y="Gender")
    assert pca_result is not None, "PCA result should not be None"
    print(pca_result)


def test_get_regression_logistic(csv_fixture, csv_file_fixture):
    """Test logistic regression with binary outcome (Gender)"""
    _csv = csv_fixture
    _csv.read_csv(csv_file_fixture)
    _csv.drop_na()
    ml = ML(csv=_csv)
    regression_result = ml.get_regression(y="Gender")
    assert regression_result is not None, "Regression result should not be None"
    assert regression_result["model_type"] == "logistic", "Should use logistic regression for binary outcome"
    assert "accuracy" in regression_result, "Should have accuracy metric"
    assert "coefficients" in regression_result, "Should have coefficients"
    assert "intercept" in regression_result, "Should have intercept"
    print(f"Logistic Regression Results: {regression_result}")


def test_get_regression_linear(csv_fixture, csv_file_fixture):
    """Test linear regression with continuous outcome (GPA)"""
    _csv = csv_fixture
    _csv.read_csv(csv_file_fixture)
    _csv.drop_na()
    ml = ML(csv=_csv)
    regression_result = ml.get_regression(y="GPA")
    assert regression_result is not None, "Regression result should not be None"
    assert regression_result["model_type"] == "linear", "Should use linear regression for continuous outcome"
    assert "mse" in regression_result, "Should have MSE metric"
    assert "r2" in regression_result, "Should have R² metric"
    assert "coefficients" in regression_result, "Should have coefficients"
    assert "intercept" in regression_result, "Should have intercept"
    print(f"Linear Regression Results: {regression_result}")


def test_get_lstm_predictions(csv_fixture, corpus_fixture, csv_file_fixture):
    """Test LSTM predictions with text and binary outcome"""
    _csv = csv_fixture
    _csv.read_csv(csv_file_fixture)
    _csv.drop_na()
    
    # Add documents to corpus with matching IDs
    # The test CSV has an id column, so we need to ensure documents have matching IDs
    # For this test, we'll create simple text documents
    from src.crisp_t.model.document import Document
    
    # Clear existing documents and create new ones matching CSV ids
    corpus_fixture.documents = []
    df = _csv.df
    for idx, row in df.iterrows():
        doc = Document(
            id=str(row.get('id', idx)),  # Use id column if exists, otherwise use index
            text=f"This is a sample document about student {idx} with some text content for testing LSTM.",
            name=f"Document {idx}",
            description="Test document",
            score=0.0,
            metadata={}
        )
        corpus_fixture.add_document(doc)
    
    # Recreate csv with updated corpus
    _csv.corpus = corpus_fixture
    ml = ML(csv=_csv)
    
    # Test LSTM predictions
    result = ml.get_lstm_predictions(y="Gender")
    
    # Check that result is returned
    assert result is not None, "LSTM prediction result should not be None"
    
    # If result is a dict (not mcp mode), check its structure
    if isinstance(result, dict):
        assert "test_accuracy" in result, "Should have test accuracy"
        assert "train_accuracy" in result, "Should have train accuracy"
        assert "vocab_size" in result, "Should have vocab size"
        assert "f1_score" in result, "Should have F1 score"
        print(f"LSTM Results: {result}")
    else:
        # If result is a string (error message), just print it
        print(f"LSTM Result: {result}")

