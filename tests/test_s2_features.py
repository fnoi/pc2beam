"""
Tests for S2Features class.
"""

import pytest
import numpy as np
from pc2beam.data import S2Features


def test_s2_features_initialization():
    """Test S2Features initialization."""
    s2_features = S2Features()
    assert len(s2_features.s2_vectors) == 0
    assert len(s2_features.line_points) == 0
    assert len(s2_features.instance_ids) == 0


def test_add_instance_features():
    """Test adding instance features."""
    s2_features = S2Features()
    
    # Add features for instance 1
    s2_vector = np.array([1.0, 0.0, 0.0])
    line_point = np.array([0.0, 0.0, 0.0])
    
    s2_features.add_instance_features(1, s2_vector, line_point)
    
    assert 1 in s2_features.s2_vectors
    assert 1 in s2_features.line_points
    assert 1 in s2_features.instance_ids
    assert np.array_equal(s2_features.get_s2_vector(1), s2_vector)
    assert np.array_equal(s2_features.get_line_point(1), line_point)


def test_has_instance():
    """Test has_instance method."""
    s2_features = S2Features()
    
    # Add features for instance 1
    s2_vector = np.array([1.0, 0.0, 0.0])
    line_point = np.array([0.0, 0.0, 0.0])
    s2_features.add_instance_features(1, s2_vector, line_point)
    
    assert s2_features.has_instance(1) is True
    assert s2_features.has_instance(2) is False


def test_get_all_instances():
    """Test get_all_instances method."""
    s2_features = S2Features()
    
    # Add features for multiple instances
    s2_vector1 = np.array([1.0, 0.0, 0.0])
    line_point1 = np.array([0.0, 0.0, 0.0])
    s2_features.add_instance_features(1, s2_vector1, line_point1)
    
    s2_vector2 = np.array([0.0, 1.0, 0.0])
    line_point2 = np.array([1.0, 1.0, 1.0])
    s2_features.add_instance_features(2, s2_vector2, line_point2)
    
    instances = s2_features.get_all_instances()
    assert len(instances) == 2
    assert 1 in instances
    assert 2 in instances


def test_to_dict():
    """Test to_dict method."""
    s2_features = S2Features()
    
    # Add features for instance 1
    s2_vector = np.array([1.0, 0.0, 0.0])
    line_point = np.array([0.0, 0.0, 0.0])
    s2_features.add_instance_features(1, s2_vector, line_point)
    
    result_dict = s2_features.to_dict()
    
    assert 1 in result_dict
    assert "s2" in result_dict[1]
    assert "line_point" in result_dict[1]
    assert np.array_equal(result_dict[1]["s2"], s2_vector)
    assert np.array_equal(result_dict[1]["line_point"], line_point)


def test_from_dict():
    """Test from_dict class method."""
    # Create test dictionary
    test_dict = {
        1: {
            "s2": np.array([1.0, 0.0, 0.0]),
            "line_point": np.array([0.0, 0.0, 0.0])
        },
        2: {
            "s2": np.array([0.0, 1.0, 0.0]),
            "line_point": np.array([1.0, 1.0, 1.0])
        }
    }
    
    s2_features = S2Features.from_dict(test_dict)
    
    assert len(s2_features.instance_ids) == 2
    assert s2_features.has_instance(1)
    assert s2_features.has_instance(2)
    assert np.array_equal(s2_features.get_s2_vector(1), test_dict[1]["s2"])
    assert np.array_equal(s2_features.get_line_point(2), test_dict[2]["line_point"])


def test_key_error():
    """Test that KeyError is raised for non-existent instances."""
    s2_features = S2Features()
    
    with pytest.raises(KeyError):
        s2_features.get_s2_vector(1)
    
    with pytest.raises(KeyError):
        s2_features.get_line_point(1) 