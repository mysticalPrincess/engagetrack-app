"""
Test engagement detection logic
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    get_engagement_state,
    calculate_iou,
    assign_student_id,
    get_pose_features,
    classify_posture,
    combine_engagement_signals,
    mid
)


class TestEngagementMapping:
    """Test engagement state mapping"""
    
    def test_engaged_states(self):
        """Test engaged state mapping"""
        assert get_engagement_state('engaged') == 'engaged'
        assert get_engagement_state('focused') == 'engaged'
    
    def test_neutral_states(self):
        """Test neutral state mapping"""
        assert get_engagement_state('neutral') == 'neutral'
        assert get_engagement_state('confused') == 'neutral'
    
    def test_bored_states(self):
        """Test bored state mapping"""
        assert get_engagement_state('bored') == 'bored'
        assert get_engagement_state('frustrated') == 'bored'
        assert get_engagement_state('looking_away') == 'bored'
        assert get_engagement_state('distracted') == 'bored'
    
    def test_sleepy_states(self):
        """Test sleepy state mapping"""
        assert get_engagement_state('sleepy') == 'sleepy'
        assert get_engagement_state('yawning') == 'sleepy'
    
    def test_unknown_state(self):
        """Test unknown state defaults to neutral"""
        assert get_engagement_state('unknown') == 'neutral'
        assert get_engagement_state('random') == 'neutral'


class TestIOUCalculation:
    """Test Intersection over Union calculation"""
    
    def test_perfect_overlap(self):
        """Test IOU with perfect overlap"""
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]
        iou = calculate_iou(box1, box2)
        assert iou == 1.0
    
    def test_no_overlap(self):
        """Test IOU with no overlap"""
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]
        iou = calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_partial_overlap(self):
        """Test IOU with partial overlap"""
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
        # Expected IOU = 2500 / 17500 ≈ 0.143
        assert 0.14 <= iou <= 0.15
    
    def test_contained_box(self):
        """Test IOU with one box inside another"""
        box1 = [0, 0, 100, 100]
        box2 = [25, 25, 75, 75]
        iou = calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
        # Smaller box area / larger box area = 0.25
        assert 0.24 <= iou <= 0.26


class TestStudentTracking:
    """Test student ID assignment and tracking"""
    
    def test_new_student_assignment(self):
        """Test assigning ID to new student"""
        import app
        app.student_trackers = {}
        app.next_student_id = 1
        
        box = [100, 100, 200, 200]
        student_id = assign_student_id(box, [])
        
        assert student_id == 1
        assert 1 in app.student_trackers
    
    def test_existing_student_tracking(self):
        """Test tracking existing student"""
        import app
        import time
        
        app.student_trackers = {
            1: {'box': [100, 100, 200, 200], 'last_seen': time.time()}
        }
        app.next_student_id = 2
        
        # Similar box should match existing student
        box = [105, 105, 205, 205]
        student_id = assign_student_id(box, [])
        
        assert student_id == 1  # Should reuse existing ID


class TestPoseFeatures:
    """Test pose feature extraction"""
    
    def test_mid_calculation(self):
        """Test midpoint calculation"""
        a = np.array([0, 0])
        b = np.array([100, 100])
        midpoint = mid(a, b)
        
        assert np.array_equal(midpoint, np.array([50, 50]))
    
    def test_pose_features_extraction(self):
        """Test extracting features from keypoints"""
        # Create mock keypoints (17, 3) - upright posture
        keypoints = np.zeros((17, 3))
        
        # Set key points with high confidence
        keypoints[0] = [320, 100, 0.9]   # nose
        keypoints[1] = [310, 95, 0.9]    # left eye
        keypoints[2] = [330, 95, 0.9]    # right eye
        keypoints[5] = [300, 150, 0.9]   # left shoulder
        keypoints[6] = [340, 150, 0.9]   # right shoulder
        keypoints[11] = [300, 250, 0.9]  # left hip
        keypoints[12] = [340, 250, 0.9]  # right hip
        
        features = get_pose_features(keypoints)
        
        assert features is not None
        assert 'torso_angle' in features
        assert 'head_angle' in features
        assert 'head_droop_normalized' in features
        assert 'head_tilt' in features
        assert 'eye_conf' in features
    
    def test_pose_features_low_confidence(self):
        """Test that low confidence keypoints return None"""
        keypoints = np.zeros((17, 3))
        # All keypoints have low confidence
        keypoints[:, 2] = 0.1
        
        features = get_pose_features(keypoints)
        assert features is None


class TestPostureClassification:
    """Test posture classification"""
    
    def test_engaged_posture(self):
        """Test engaged posture classification"""
        features = {
            'torso_angle': 5.0,      # Upright
            'head_droop_normalized': 0.02,  # Head up
            'head_tilt': 0.03,       # Minimal tilt
            'eye_conf': 0.8          # Good eye confidence
        }
        
        posture = classify_posture(features)
        assert posture == 'engaged'
    
    def test_sleepy_posture(self):
        """Test sleepy posture classification"""
        features = {
            'torso_angle': 20.0,     # Hunched forward
            'head_droop_normalized': 0.20,  # Head drooped
            'head_tilt': 0.05,
            'eye_conf': 0.3          # Low eye confidence
        }
        
        posture = classify_posture(features)
        assert posture == 'sleepy'
    
    def test_bored_posture(self):
        """Test bored posture classification"""
        features = {
            'torso_angle': 25.0,     # Slouched
            'head_droop_normalized': 0.10,
            'head_tilt': 0.05,
            'eye_conf': 0.6
        }
        
        posture = classify_posture(features)
        assert posture == 'bored'
    
    def test_yawning_posture(self):
        """Test yawning posture classification"""
        features = {
            'torso_angle': -12.0,    # Leaning back
            'head_droop_normalized': -0.15,  # Head tilted back
            'head_tilt': 0.04,
            'eye_conf': 0.5
        }
        
        posture = classify_posture(features)
        assert posture == 'yawning'
    
    def test_confused_posture(self):
        """Test confused posture classification"""
        features = {
            'torso_angle': 8.0,
            'head_droop_normalized': 0.03,
            'head_tilt': 0.10,       # Moderate head tilt
            'eye_conf': 0.7
        }
        
        posture = classify_posture(features)
        assert posture == 'confused'
    
    def test_frustrated_posture(self):
        """Test frustrated posture classification"""
        features = {
            'torso_angle': 10.0,
            'head_droop_normalized': 0.05,
            'head_tilt': 0.18,       # Extreme head tilt
            'eye_conf': 0.6
        }
        
        posture = classify_posture(features)
        assert posture == 'frustrated'


class TestSignalCombination:
    """Test combining face and posture signals"""
    
    def test_both_agree_engaged(self):
        """Test when both signals agree on engaged"""
        result = combine_engagement_signals('engaged', 'engaged')
        assert result == 'engaged'
    
    def test_both_agree_sleepy(self):
        """Test when both signals agree on sleepy"""
        result = combine_engagement_signals('sleepy', 'sleepy')
        assert result == 'sleepy'
    
    def test_face_priority(self):
        """Test that face detection has priority"""
        result = combine_engagement_signals('engaged', 'bored')
        # Face says engaged, posture says bored - should trust face
        assert result == 'engaged'
    
    def test_sleepy_override(self):
        """Test sleepy state overrides"""
        result = combine_engagement_signals('engaged', 'sleepy')
        assert result == 'sleepy'
        
        result = combine_engagement_signals('sleepy', 'engaged')
        assert result == 'sleepy'
    
    def test_yawning_override(self):
        """Test yawning state overrides"""
        result = combine_engagement_signals('engaged', 'yawning')
        assert result == 'yawning'
    
    def test_no_posture_state(self):
        """Test when posture state is None"""
        result = combine_engagement_signals('engaged', None)
        assert result == 'engaged'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
