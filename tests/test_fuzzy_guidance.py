import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fuzzy_guidance import FuzzyGuidanceSystem

@pytest.fixture
def fis():
    return FuzzyGuidanceSystem()

def test_stable_high_confidence_center(fis):
    # Test 1
    result = fis.compute(angle_offset=0.0, confidence=0.9, stability=0.9)
    assert result['direction'] == 'center'
    # Expect High Urgency, Fast Frequency
    assert result['urgency'] > 0.7
    assert result['frequency'] > 0.7
    assert result['should_announce'] is True

def test_stable_high_confidence_left(fis):
    # Test 2
    result = fis.compute(angle_offset=-0.8, confidence=0.9, stability=0.9)
    assert result['direction'] == 'left'
    assert result['urgency'] > 0.7
    assert result['frequency'] > 0.7
    assert result['should_announce'] is True

def test_stable_high_confidence_right(fis):
    # Test 3
    result = fis.compute(angle_offset=0.8, confidence=0.9, stability=0.9)
    assert result['direction'] == 'right'
    assert result['urgency'] > 0.7
    assert result['frequency'] > 0.7
    assert result['should_announce'] is True

def test_medium_confidence(fis):
    # Test 4
    result = fis.compute(angle_offset=0.0, confidence=0.5, stability=0.9)
    assert result['direction'] == 'center'
    # Medium confidence, Stable -> Medium urgency, Slow frequency
    assert 0.3 <= result['urgency'] <= 0.7
    assert result['frequency'] < 0.5
    assert result['should_announce'] is True

def test_low_confidence(fis):
    # Test 5
    result = fis.compute(angle_offset=0.0, confidence=0.1, stability=0.9)
    assert result['direction'] == 'center'
    # Low confidence -> Low urgency, Slow frequency
    assert result['urgency'] < 0.4
    assert result['frequency'] < 0.4
    assert result['should_announce'] is False # Due to confidence < 0.30

def test_high_confidence_unstable(fis):
    # Test 6
    result_stable = fis.compute(angle_offset=0.0, confidence=0.9, stability=0.9)
    result_unstable = fis.compute(angle_offset=0.0, confidence=0.9, stability=0.1)
    
    # Unstable should reduce urgency compared to stable
    assert result_unstable['urgency'] < result_stable['urgency']
    # Rule says: High conf + Unstable -> Medium urgency, Slow frequency
    assert 0.3 <= result_unstable['urgency'] <= 0.7
    assert result_unstable['frequency'] < 0.5

def test_low_medium_confidence_unstable(fis):
    # Test 7
    result_med = fis.compute(angle_offset=0.0, confidence=0.5, stability=0.1)
    result_low = fis.compute(angle_offset=0.0, confidence=0.2, stability=0.1)
    
    # Unstable + Low/Medium -> Low urgency, Slow frequency
    assert result_med['urgency'] < 0.4
    assert result_med['frequency'] < 0.4
    assert result_low['urgency'] < 0.4
    assert result_low['frequency'] < 0.4

def test_direction_sweep(fis):
    # Test 8
    result_left = fis.compute(angle_offset=-0.9, confidence=0.8, stability=0.8)
    assert result_left['direction'] == 'left'
    
    result_center = fis.compute(angle_offset=0.0, confidence=0.8, stability=0.8)
    assert result_center['direction'] == 'center'
    
    result_right = fis.compute(angle_offset=0.9, confidence=0.8, stability=0.8)
    assert result_right['direction'] == 'right'

def test_boundary_inputs(fis):
    # Testing extremes
    res_1 = fis.compute(angle_offset=-1.0, confidence=0.0, stability=0.0)
    assert res_1['direction'] == 'left'
    assert res_1['urgency'] >= 0.0
    
    res_2 = fis.compute(angle_offset=1.0, confidence=1.0, stability=1.0)
    assert res_2['direction'] == 'right'
    assert res_2['urgency'] <= 1.0

def test_missing_or_invalid_inputs(fis):
    # Out of bounds should be clipped automatically (based on implementation)
    res_clip = fis.compute(angle_offset=-1.5, confidence=-0.5, stability=1.5)
    assert res_clip['direction'] == 'left'
    # -0.5 is clipped to 0 (low conf) -> low urgency
    assert res_clip['urgency'] < 0.4
