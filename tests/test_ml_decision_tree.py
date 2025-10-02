import logging

from src.crisp_t.ml import ML

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_get_decision_tree_classes(csv_fixture, csv_file_fixture):
    # Prepare CSV data
    csv_fixture.read_csv(csv_file_fixture)
    csv_fixture.drop_na()

    # Initialize ML with prepared CSV
    ml = ML(csv=csv_fixture)

    # Execute decision tree classification
    cm, importance = ml.get_decision_tree_classes(y="Gender")

    # Assertions: confusion matrix shape and importance vector length
    assert cm is not None, "Confusion matrix should not be None"
    assert (
        cm.ndim == 2 and cm.shape[0] == cm.shape[1]
    ), "Confusion matrix should be square"
    assert importance is not None, "Feature importance should not be None"
    # Importance length should match number of features used by the model
    X_np, _, _, _ = ml._process_xy(y="Gender")
    assert (
        len(importance) == X_np.shape[1]
    ), "Feature importance length must match number of features"
