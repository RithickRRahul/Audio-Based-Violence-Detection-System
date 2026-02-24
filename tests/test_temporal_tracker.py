import pytest
from src.models.temporal_tracker import RuleBasedTemporalTracker

def test_spike_detection():
    tracker = RuleBasedTemporalTracker(window_size=5)
    
    # Simulate a sudden spike from quiet safe scores to high danger
    inputs = [0.1, 0.2, 0.15, 0.95]
    for score in inputs:
        result = tracker.update(score)
        
    assert result["trend"] == "spike"
    assert result["prediction"] == "violence likely to escalate"
    # Escalation scores shouldn't drop the context below the highest trigger
    assert result["escalation_score"] >= 0.95

def test_rising_trend():
    tracker = RuleBasedTemporalTracker(window_size=5)
    
    # Simulate a steady, concerning increase over 5 chunks
    inputs = [0.2, 0.35, 0.5, 0.65, 0.8]
    for score in inputs:
        result = tracker.update(score)
        
    assert result["trend"] == "rising"
    import math
    assert math.isclose(result["escalation_score"], 0.8, rel_tol=1e-5) or result["escalation_score"] >= 0.8

def test_sustained_aggression():
    tracker = RuleBasedTemporalTracker(window_size=5)
    
    # Simulate 3+ chunks hovering around 0.6 (above 0.5 threshold)
    inputs = [0.6, 0.55, 0.65, 0.6, 0.62]
    for score in inputs:
        result = tracker.update(score)
        
    assert result["trend"] == "sustained"

def test_falling_trend():
    tracker = RuleBasedTemporalTracker(window_size=5)
    
    # Simulate an incident that is de-escalating
    inputs = [0.9, 0.7, 0.5, 0.3, 0.1]
    for score in inputs:
        result = tracker.update(score)
        
    assert result["trend"] == "falling"

def test_stable_safe():
    tracker = RuleBasedTemporalTracker(window_size=5)
    
    # Simulate a boring, safe room
    inputs = [0.1, 0.15, 0.12, 0.1, 0.11]
    for score in inputs:
        result = tracker.update(score)
        
    assert result["trend"] == "stable"
    assert result["escalation_score"] < 0.5
