import json
import logging
import pytest
import pandas as pd
import numpy as np

from src.crisp_t.tdabm import Tdabm
from src.crisp_t.model.corpus import Corpus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def tdabm_corpus():
    """Create a test corpus with numeric data for TDABM."""
    # Create a DataFrame with continuous and ordinal variables
    np.random.seed(42)
    n_points = 50
    
    df = pd.DataFrame({
        'x1': np.random.uniform(0, 10, n_points),
        'x2': np.random.uniform(0, 10, n_points),
        'x3': np.random.uniform(0, 10, n_points),
        'y_continuous': np.random.uniform(0, 100, n_points),
        'y_binary': np.random.choice([0, 1], n_points),
        'categorical': np.random.choice(['A', 'B', 'C'], n_points)
    })
    
    corpus = Corpus(
        id="tdabm_test",
        name="TDABM Test Corpus",
        description="Test corpus for TDABM analysis",
        df=df
    )
    
    return corpus


def test_tdabm_initialization(tdabm_corpus):
    """Test TDABM initialization with valid corpus."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    assert tdabm.corpus == tdabm_corpus
    assert tdabm.corpus.df is not None


def test_tdabm_initialization_without_dataframe():
    """Test TDABM initialization fails without DataFrame."""
    corpus = Corpus(
        id="no_df",
        name="No DataFrame Corpus",
        description="Corpus without DataFrame"
    )
    
    with pytest.raises(ValueError, match="Corpus must have a DataFrame"):
        Tdabm(corpus=corpus)


def test_validate_variables_continuous_y(tdabm_corpus):
    """Test that y variable must be continuous (not binary)."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    # Should raise error for binary y variable
    with pytest.raises(ValueError, match="appears to be binary"):
        tdabm._validate_variables('y_binary', ['x1', 'x2'])


def test_validate_variables_missing_y(tdabm_corpus):
    """Test validation fails for missing y variable."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    with pytest.raises(ValueError, match="not found in DataFrame"):
        tdabm._validate_variables('missing_var', ['x1', 'x2'])


def test_validate_variables_missing_x(tdabm_corpus):
    """Test validation fails for missing X variable."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    with pytest.raises(ValueError, match="not found in DataFrame"):
        tdabm._validate_variables('y_continuous', ['x1', 'missing_x'])


def test_validate_variables_non_numeric_x(tdabm_corpus):
    """Test validation fails for non-numeric X variable."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    with pytest.raises(ValueError, match="must be numeric"):
        tdabm._validate_variables('y_continuous', ['x1', 'categorical'])


def test_normalize_data(tdabm_corpus):
    """Test data normalization to [0, 1] scale."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    df = tdabm.corpus.df[['x1', 'x2', 'y_continuous']]
    normalized = tdabm._normalize_data(df, ['x1', 'x2', 'y_continuous'])
    
    # Check all values are in [0, 1]
    assert normalized['x1'].min() >= 0
    assert normalized['x1'].max() <= 1
    assert normalized['x2'].min() >= 0
    assert normalized['x2'].max() <= 1
    assert normalized['y_continuous'].min() >= 0
    assert normalized['y_continuous'].max() <= 1
    
    # Check min and max are approximately 0 and 1
    assert abs(normalized['x1'].min() - 0.0) < 1e-10
    assert abs(normalized['x1'].max() - 1.0) < 1e-10


def test_normalize_constant_column(tdabm_corpus):
    """Test normalization of constant column."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    # Create DataFrame with constant column
    df = pd.DataFrame({'const': [5.0, 5.0, 5.0]})
    normalized = tdabm._normalize_data(df, ['const'])
    
    # Should be set to 0.5 for constant values
    assert all(normalized['const'] == 0.5)


def test_euclidean_distance(tdabm_corpus):
    """Test Euclidean distance calculation."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    point1 = np.array([0, 0, 0])
    point2 = np.array([1, 1, 1])
    
    distance = tdabm._euclidean_distance(point1, point2)
    expected = np.sqrt(3)
    
    assert abs(distance - expected) < 1e-10


def test_generate_tdabm_basic(tdabm_corpus):
    """Test basic TDABM generation."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    result = tdabm.generate_tdabm(
        y='y_continuous',
        x_variables='x1,x2',
        radius=0.3
    )
    
    # Check that metadata was stored
    assert 'tdabm' in tdabm.corpus.metadata
    metadata = tdabm.corpus.metadata['tdabm']
    
    # Check metadata structure
    assert metadata['y_variable'] == 'y_continuous'
    assert metadata['x_variables'] == ['x1', 'x2']
    assert metadata['radius'] == 0.3
    assert 'num_landmarks' in metadata
    assert 'landmarks' in metadata
    assert metadata['num_landmarks'] > 0
    
    # Check result is a string summary
    assert isinstance(result, str)
    assert 'TDABM Analysis Complete' in result


def test_generate_tdabm_landmarks_structure(tdabm_corpus):
    """Test that landmarks have correct structure."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    tdabm.generate_tdabm(
        y='y_continuous',
        x_variables='x1,x2,x3',
        radius=0.5
    )
    
    metadata = tdabm.corpus.metadata['tdabm']
    landmarks = metadata['landmarks']
    
    # Check at least one landmark
    assert len(landmarks) > 0
    
    # Check first landmark structure
    landmark = landmarks[0]
    assert 'id' in landmark
    assert 'location' in landmark
    assert 'point_indices' in landmark
    assert 'count' in landmark
    assert 'mean_y' in landmark
    assert 'connections' in landmark
    
    # Check types
    assert isinstance(landmark['id'], str)
    assert isinstance(landmark['location'], list)
    assert isinstance(landmark['point_indices'], list)
    assert isinstance(landmark['count'], int)
    assert isinstance(landmark['mean_y'], float)
    assert isinstance(landmark['connections'], list)


def test_generate_tdabm_coverage(tdabm_corpus):
    """Test that all points are covered by balls."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    tdabm.generate_tdabm(
        y='y_continuous',
        x_variables='x1,x2',
        radius=0.5
    )
    
    metadata = tdabm.corpus.metadata['tdabm']
    landmarks = metadata['landmarks']
    
    # Get all covered points
    all_covered = set()
    for landmark in landmarks:
        all_covered.update(landmark['point_indices'])
    
    # Should cover all points
    assert len(all_covered) == metadata['num_points']


def test_generate_tdabm_connections(tdabm_corpus):
    """Test that connections are properly identified."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    # Use smaller radius to get more landmarks
    tdabm.generate_tdabm(
        y='y_continuous',
        x_variables='x1,x2',
        radius=0.2
    )
    
    metadata = tdabm.corpus.metadata['tdabm']
    landmarks = metadata['landmarks']
    
    # With small radius, should have multiple landmarks
    assert len(landmarks) > 1
    
    # Check that connections are symmetric
    for landmark in landmarks:
        for connected_id in landmark['connections']:
            # Find the connected landmark
            connected_landmark = next(
                (lm for lm in landmarks if lm['id'] == connected_id),
                None
            )
            assert connected_landmark is not None
            # Check that the connection is symmetric
            assert landmark['id'] in connected_landmark['connections']


def test_generate_tdabm_mcp_mode(tdabm_corpus):
    """Test TDABM generation with MCP mode."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    result = tdabm.generate_tdabm(
        y='y_continuous',
        x_variables='x1,x2',
        radius=0.3,
        mcp=True
    )
    
    # Should return JSON string
    assert isinstance(result, str)
    
    # Should be valid JSON
    parsed = json.loads(result)
    assert 'y_variable' in parsed
    assert 'landmarks' in parsed


def test_generate_tdabm_different_radii(tdabm_corpus):
    """Test TDABM with different radius values."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    # Large radius should result in fewer landmarks
    tdabm.generate_tdabm(
        y='y_continuous',
        x_variables='x1,x2',
        radius=0.8
    )
    large_radius_landmarks = len(tdabm.corpus.metadata['tdabm']['landmarks'])
    
    # Small radius should result in more landmarks
    tdabm.generate_tdabm(
        y='y_continuous',
        x_variables='x1,x2',
        radius=0.2
    )
    small_radius_landmarks = len(tdabm.corpus.metadata['tdabm']['landmarks'])
    
    assert small_radius_landmarks > large_radius_landmarks


def test_generate_tdabm_single_x_variable(tdabm_corpus):
    """Test TDABM with single X variable."""
    tdabm = Tdabm(corpus=tdabm_corpus)
    
    result = tdabm.generate_tdabm(
        y='y_continuous',
        x_variables='x1',
        radius=0.3
    )
    
    assert 'tdabm' in tdabm.corpus.metadata
    metadata = tdabm.corpus.metadata['tdabm']
    assert metadata['x_variables'] == ['x1']


def test_generate_tdabm_with_nan_values():
    """Test TDABM handles NaN values correctly."""
    # Create corpus with some NaN values
    df = pd.DataFrame({
        'x1': [1, 2, np.nan, 4, 5],
        'x2': [5, 4, 3, np.nan, 1],
        'y': [10, 20, 30, 40, 50]
    })
    
    corpus = Corpus(
        id="nan_test",
        name="NaN Test Corpus",
        df=df
    )
    
    tdabm = Tdabm(corpus=corpus)
    
    result = tdabm.generate_tdabm(
        y='y',
        x_variables='x1,x2',
        radius=0.5
    )
    
    # Should only use valid (non-NaN) rows
    metadata = corpus.metadata['tdabm']
    assert metadata['num_points'] == 3  # Only 3 rows have all values
