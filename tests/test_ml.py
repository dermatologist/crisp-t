import logging

from pkg_resources import resource_filename

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


def test_get_kmeans(csv_fixture):
    folder_path = resource_filename("src.crisp_t.resources", "food_coded.csv")

    csv_fixture.read_csv(folder_path)
    csv_fixture.drop_na()
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


def test_profile(csv_fixture):

    folder_path = resource_filename("src.crisp_t.resources", "food_coded.csv")
    _csv = csv_fixture
    _csv.read_csv(folder_path)
    _csv.drop_na()
    print(_csv)
    ml = ML(csv=_csv)
    kmeans, members = ml.get_kmeans(number_of_clusters=5)
    profile = ml.profile(members, number_of_clusters=5)
    print(profile)
    assert profile is not None, "Profile should not be None"


def test_get_nnet_predictions(csv_fixture):
    folder_path = resource_filename("src.crisp_t.resources", "food_coded.csv")
    _csv = csv_fixture
    _csv.read_csv(folder_path)
    _csv.drop_na()
    ml = ML(csv=_csv)
    predictions = ml.get_nnet_predictions(y="Gender")
    assert predictions is not None, "Predictions should not be None"


def test_svm_confusion_matrix(csv_fixture):
    folder_path = resource_filename("src.crisp_t.resources", "food_coded.csv")
    _csv = csv_fixture
    _csv.read_csv(folder_path)
    _csv.drop_na()
    ml = ML(csv=_csv)
    confusion_matrix = ml.svm_confusion_matrix(y="Gender")
    assert confusion_matrix is not None, "Confusion matrix should not be None"
    human_readable = ml.format_confusion_matrix_to_human_readable(confusion_matrix)
    print(human_readable)

def test_knn_search(csv_fixture):
    folder_path = resource_filename("src.crisp_t.resources", "food_coded.csv")
    _csv = csv_fixture
    _csv.read_csv(folder_path)
    _csv.drop_na()
    ml = ML(csv=_csv)
    dist, ind = ml.knn_search(y="Gender", n=3, r=3)
    assert ind is not None, "Neighbors should not be None"
    print(f"KNN search for Gender (n=3, record no 3): {ind} with distances {dist}")


def test_get_xgb_classes(csv_fixture):
    folder_path = resource_filename("src.crisp_t.resources", "food_coded.csv")
    _csv = csv_fixture
    _csv.read_csv(folder_path)
    _csv.drop_na()
    ml = ML(csv=_csv)
    xgb_classes = ml.get_xgb_classes(y="Gender")
    assert xgb_classes is not None, "XGBoost classes should not be None"
    human_readable = ml.format_confusion_matrix_to_human_readable(xgb_classes)
    print(human_readable)
