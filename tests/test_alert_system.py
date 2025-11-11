"""
Test real-time alert system
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import check_alert_trigger, update_active_alerts


class TestAlertTriggers:
    """Test alert triggering logic"""
    
    def setup_method(self):
        """Reset alert tracking before each test"""
        import app
        app.previous_states = {}
        app.alert_cooldown = {}
        app.active_alerts = []
    
    def test_engaged_to_sleepy_triggers_alert(self):
        """Test alert triggers on engaged -> sleepy transition"""
        should_alert, message = check_alert_trigger(1, 'engaged')
        assert should_alert is False  # First state, no alert
        
        should_alert, message = check_alert_trigger(1, 'sleepy')
        assert should_alert is True
        assert 'SLEEPY' in message
        assert 'Student #1' in message
    
    def test_engaged_to_bored_triggers_alert(self):
        """Test alert triggers on engaged -> bored transition"""
        check_alert_trigger(1, 'engaged')
        should_alert, message = check_alert_trigger(1, 'bored')
        
        assert should_alert is True
        assert 'BORED' in message
    
    def test_engaged_to_yawning_triggers_alert(self):
        """Test alert triggers on engaged -> yawning transition"""
        check_alert_trigger(1, 'engaged')
        should_alert, message = check_alert_trigger(1, 'yawning')
        
        assert should_alert is True
        assert 'YAWNING' in message
    
    def test_engaged_to_frustrated_triggers_alert(self):
        """Test alert triggers on engaged -> frustrated transition"""
        check_alert_trigger(1, 'engaged')
        should_alert, message = check_alert_trigger(1, 'frustrated')
        
        assert should_alert is True
        assert 'FRUSTRATED' in message
    
    def test_neutral_to_sleepy_triggers_alert(self):
        """Test alert triggers from neutral state"""
        check_alert_trigger(1, 'neutral')
        should_alert, message = check_alert_trigger(1, 'sleepy')
        
        assert should_alert is True
    
    def test_sleepy_to_sleepy_no_alert(self):
        """Test no alert when staying in same disengaged state"""
        check_alert_trigger(1, 'engaged')
        check_alert_trigger(1, 'sleepy')  # First alert
        
        should_alert, message = check_alert_trigger(1, 'sleepy')
        assert should_alert is False  # No second alert
    
    def test_engaged_to_engaged_no_alert(self):
        """Test no alert when staying engaged"""
        check_alert_trigger(1, 'engaged')
        should_alert, message = check_alert_trigger(1, 'engaged')
        
        assert should_alert is False
    
    def test_sleepy_to_engaged_no_alert(self):
        """Test no alert on recovery (positive change)"""
        check_alert_trigger(1, 'sleepy')
        should_alert, message = check_alert_trigger(1, 'engaged')
        
        assert should_alert is False  # Recovery is good, no alert
    
    def test_bored_to_engaged_no_alert(self):
        """Test no alert on recovery from bored"""
        check_alert_trigger(1, 'bored')
        should_alert, message = check_alert_trigger(1, 'engaged')
        
        assert should_alert is False


class TestAlertCooldown:
    """Test alert cooldown mechanism"""
    
    def setup_method(self):
        """Reset alert tracking before each test"""
        import app
        app.previous_states = {}
        app.alert_cooldown = {}
        app.active_alerts = []
    
    def test_cooldown_prevents_spam(self):
        """Test cooldown prevents repeated alerts"""
        # First transition: engaged -> sleepy
        check_alert_trigger(1, 'engaged')
        should_alert, _ = check_alert_trigger(1, 'sleepy')
        assert should_alert is True
        
        # Recover and try again immediately
        check_alert_trigger(1, 'engaged')
        should_alert, _ = check_alert_trigger(1, 'bored')
        assert should_alert is False  # Cooldown active
    
    def test_different_students_independent_cooldown(self):
        """Test different students have independent cooldowns"""
        # Student 1 alert
        check_alert_trigger(1, 'engaged')
        should_alert1, _ = check_alert_trigger(1, 'sleepy')
        assert should_alert1 is True
        
        # Student 2 should still alert
        check_alert_trigger(2, 'engaged')
        should_alert2, _ = check_alert_trigger(2, 'sleepy')
        assert should_alert2 is True
    
    def test_cooldown_expires(self):
        """Test cooldown expires after 10 seconds"""
        import app
        
        # Trigger alert
        check_alert_trigger(1, 'engaged')
        check_alert_trigger(1, 'sleepy')
        
        # Manually expire cooldown
        app.alert_cooldown[1] = time.time() - 11  # 11 seconds ago
        
        # Should allow new alert
        check_alert_trigger(1, 'engaged')
        should_alert, _ = check_alert_trigger(1, 'bored')
        assert should_alert is True


class TestActiveAlerts:
    """Test active alerts management"""
    
    def setup_method(self):
        """Reset alert tracking before each test"""
        import app
        app.previous_states = {}
        app.alert_cooldown = {}
        app.active_alerts = []
    
    def test_alert_added_to_active_list(self):
        """Test alert is added to active list"""
        import app
        
        check_alert_trigger(1, 'engaged')
        should_alert, message = check_alert_trigger(1, 'sleepy')
        
        if should_alert:
            app.active_alerts.append((message, time.time()))
        
        assert len(app.active_alerts) == 1
        assert 'SLEEPY' in app.active_alerts[0][0]
    
    def test_update_removes_old_alerts(self):
        """Test old alerts are removed"""
        import app
        
        # Add old alert (6 seconds ago)
        app.active_alerts.append(('Old alert', time.time() - 6))
        
        # Add recent alert
        app.active_alerts.append(('Recent alert', time.time()))
        
        update_active_alerts()
        
        # Only recent alert should remain
        assert len(app.active_alerts) == 1
        assert 'Recent' in app.active_alerts[0][0]
    
    def test_multiple_alerts_display(self):
        """Test multiple alerts can be active"""
        import app
        
        current_time = time.time()
        app.active_alerts = [
            ('Student #1 appears SLEEPY', current_time),
            ('Student #2 appears BORED', current_time),
            ('Student #3 is YAWNING', current_time)
        ]
        
        update_active_alerts()
        
        assert len(app.active_alerts) == 3


class TestAlertMessages:
    """Test alert message formatting"""
    
    def setup_method(self):
        """Reset alert tracking before each test"""
        import app
        app.previous_states = {}
        app.alert_cooldown = {}
        app.active_alerts = []
    
    def test_sleepy_message_format(self):
        """Test sleepy alert message"""
        check_alert_trigger(1, 'engaged')
        _, message = check_alert_trigger(1, 'sleepy')
        
        assert message == 'Student #1 appears SLEEPY'
    
    def test_bored_message_format(self):
        """Test bored alert message"""
        check_alert_trigger(2, 'engaged')
        _, message = check_alert_trigger(2, 'bored')
        
        assert message == 'Student #2 appears BORED'
    
    def test_yawning_message_format(self):
        """Test yawning alert message"""
        check_alert_trigger(3, 'engaged')
        _, message = check_alert_trigger(3, 'yawning')
        
        assert message == 'Student #3 is YAWNING'
    
    def test_frustrated_message_format(self):
        """Test frustrated alert message"""
        check_alert_trigger(4, 'engaged')
        _, message = check_alert_trigger(4, 'frustrated')
        
        assert message == 'Student #4 appears FRUSTRATED'


class TestEdgeCases:
    """Test edge cases in alert system"""
    
    def setup_method(self):
        """Reset alert tracking before each test"""
        import app
        app.previous_states = {}
        app.alert_cooldown = {}
        app.active_alerts = []
    
    def test_first_detection_no_alert(self):
        """Test first detection doesn't trigger alert"""
        should_alert, _ = check_alert_trigger(1, 'sleepy')
        assert should_alert is False  # No previous state
    
    def test_confused_to_bored_triggers_alert(self):
        """Test transition from confused (engaged) to bored"""
        check_alert_trigger(1, 'confused')
        should_alert, _ = check_alert_trigger(1, 'bored')
        
        assert should_alert is True
    
    def test_bored_to_frustrated_no_alert(self):
        """Test transition between disengaged states"""
        check_alert_trigger(1, 'bored')
        should_alert, _ = check_alert_trigger(1, 'frustrated')
        
        assert should_alert is False  # Both disengaged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
