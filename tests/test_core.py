"""
Tests for core functionality.
"""

import pytest
from pc2beam.core import PointCloudProcessor, BeamModelGenerator

def test_point_cloud_processor_initialization():
    """Test PointCloudProcessor initialization."""
    processor = PointCloudProcessor()
    assert processor.point_cloud is None

def test_beam_model_generator_initialization():
    """Test BeamModelGenerator initialization."""
    generator = BeamModelGenerator()
    assert generator.processed_data is None 