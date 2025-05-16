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
    kmeans, centroids = ml.get_kmeans(c=5)
    # print(centroids)
    # assert kmeans is not None
    # assert centroids is not None
