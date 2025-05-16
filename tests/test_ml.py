import logging

from pkg_resources import resource_filename
from src.crisp_t.ml import ML
from src.crisp_t.csv import Csv

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ml_initialization(corpus_fixture):
    ml = ML(
        corpus=corpus_fixture,
    )
    assert ml._corpus == corpus_fixture, "Corpus should be set correctly"


def test_get_kmeans(corpus_fixture):
    ml = ML(
        corpus=corpus_fixture,
    )
    folder_path = resource_filename("src.crisp_t.resources", "food_coded.csv")
    csv = Csv(
        corpus=corpus_fixture,
        comma_separated_text_columns="comfort_food,comfort_food_reasons,diet_current",
    )
    csv.read_csv(folder_path)
    csv.drop_na()
    ml.csv = csv
    kmeans, members = ml.get_kmeans(number_of_clusters=5)
    print(kmeans)
    print(members)
    assert kmeans is not None, "KMeans clustering should not be None"
    assert members is not None, "Members should not be None"
    # [1 1 1 0 0 2 3 1 1 0 1 3 4]
    # [[3, 4, 9], [0, 1, 2, 7, 8, 10], [5], [6, 11], [12]]


def test_profile(corpus_fixture):
    ml = ML(
        corpus=corpus_fixture,
    )
    folder_path = resource_filename("src.crisp_t.resources", "food_coded.csv")
    csv = Csv(
        corpus=corpus_fixture,
        comma_separated_text_columns="comfort_food,comfort_food_reasons,diet_current",
    )
    csv.read_csv(folder_path)
    csv.drop_na()
    ml.csv = csv
    kmeans, members = ml.get_kmeans(number_of_clusters=5)
    profile = ml.profile(members, number_of_clusters=5)
    print(profile)
    assert profile is not None, "Profile should not be None"
