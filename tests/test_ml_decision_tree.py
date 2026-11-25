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


def test_decision_tree_classifier(csv_fixture, csv_file_fixture):
    """Test the new decision_tree_classifier method with actual Decision Tree."""
    # Prepare CSV data
    csv_fixture.read_csv(csv_file_fixture)
    csv_fixture.drop_na()

    # Initialize ML with prepared CSV
    ml = ML(csv=csv_fixture)

    # Execute decision tree classifier (actual tree, not Random Forest)
    cm, importance, tree_rules, tree_model = ml.decision_tree_classifier(
        y="Gender", max_depth=5, min_samples_split=2
    )

    # Assertions
    assert cm is not None, "Confusion matrix should not be None"
    assert (
        cm.ndim == 2 and cm.shape[0] == cm.shape[1]
    ), "Confusion matrix should be square"
    assert importance is not None, "Feature importance should not be None"
    assert tree_rules is not None, "Tree rules should not be None"
    assert isinstance(tree_rules, str), "Tree rules should be a string"
    assert tree_model is not None, "Tree model should not be None"

    # Check that tree rules contain expected content
    assert len(tree_rules) > 0, "Tree rules should not be empty"
    assert (
        "|--- " in tree_rules or "class:" in tree_rules
    ), "Tree rules should contain decision tree structure"

    # Check tree depth
    assert tree_model.get_depth() <= 5, "Tree depth should respect max_depth parameter"

    # Check feature importance length matches number of features
    X_np, _, _, _ = ml._process_xy(y="Gender")
    assert (
        len(importance) == X_np.shape[1]
    ), "Feature importance length must match number of features"


def test_decision_tree_classifier_with_max_depth(csv_fixture, csv_file_fixture):
    """Test decision tree with different max_depth values."""
    csv_fixture.read_csv(csv_file_fixture)
    csv_fixture.drop_na()
    ml = ML(csv=csv_fixture)

    # Test with max_depth=3
    cm1, _, _, tree_model1 = ml.decision_tree_classifier(y="Gender", max_depth=3)
    assert tree_model1.get_depth() <= 3, "Tree depth should be at most 3"

    # Test with max_depth=10
    cm2, _, _, tree_model2 = ml.decision_tree_classifier(y="Gender", max_depth=10)
    assert tree_model2.get_depth() <= 10, "Tree depth should be at most 10"

    # Deeper tree should have more or equal depth
    assert (
        tree_model2.get_depth() >= tree_model1.get_depth()
    ), "Deeper max_depth should result in equal or deeper tree"


def test_decision_tree_classifier_mcp_output(csv_fixture, csv_file_fixture):
    """Test decision tree classifier with MCP output format."""
    csv_fixture.read_csv(csv_file_fixture)
    csv_fixture.drop_na()
    ml = ML(csv=csv_fixture)

    # Execute with mcp=True
    result = ml.decision_tree_classifier(y="Gender", max_depth=5, mcp=True)

    # Assertions for MCP output
    assert result is not None, "MCP result should not be None"
    assert isinstance(result, str), "MCP result should be a string"
    assert "Confusion Matrix" in result, "MCP result should contain confusion matrix"
    assert "Accuracy" in result, "MCP result should contain accuracy"
    assert "Decision Tree Rules" in result, "MCP result should contain tree rules"
    assert "Tree Depth" in result, "MCP result should contain tree depth"
