"""
Test script for the real-time alert system
Simulates state transitions to verify alert triggering logic
"""

import time
from collections import defaultdict

# Simulate the alert tracking globals
previous_states = {}
alert_cooldown = {}
active_alerts = []

def check_alert_trigger(student_id, current_state):
    """
    Check if an alert should be triggered for state change.
    Only alerts when transitioning from engaged/neutral to disengaged states.
    Returns: (should_alert: bool, alert_message: str or None)
    """
    global previous_states, alert_cooldown
    
    current_time = time.time()
    
    # Define disengaged states that should trigger alerts
    disengaged_states = ['sleepy', 'bored', 'yawning', 'frustrated']
    engaged_states = ['engaged', 'neutral', 'confused']
    
    # Check cooldown (don't alert same student within 10 seconds)
    if student_id in alert_cooldown:
        if current_time - alert_cooldown[student_id] < 10.0:
            return False, None
    
    # Get previous state
    prev_state = previous_states.get(student_id, 'engaged')
    
    # Update previous state
    previous_states[student_id] = current_state
    
    # Alert only on transition from engaged/neutral to disengaged
    if prev_state in engaged_states and current_state in disengaged_states:
        alert_cooldown[student_id] = current_time
        
        # Create specific alert message
        alert_messages = {
            'sleepy': f'Student #{student_id} appears SLEEPY',
            'bored': f'Student #{student_id} appears BORED',
            'yawning': f'Student #{student_id} is YAWNING',
            'frustrated': f'Student #{student_id} appears FRUSTRATED'
        }
        
        return True, alert_messages.get(current_state, f'Student #{student_id} disengaged')
    
    return False, None

# Test scenarios
print("=" * 60)
print("Testing Real-Time Alert System")
print("=" * 60)

# Test 1: Engaged -> Sleepy (should alert)
print("\n[Test 1] Student 1: engaged -> sleepy")
should_alert, msg = check_alert_trigger(1, 'engaged')
print(f"  First frame (engaged): Alert={should_alert}, Message={msg}")
should_alert, msg = check_alert_trigger(1, 'sleepy')
print(f"  Second frame (sleepy): Alert={should_alert}, Message={msg}")
assert should_alert == True, "Should alert on engaged->sleepy transition"

# Test 2: Sleepy -> Sleepy (should NOT alert - already disengaged)
print("\n[Test 2] Student 1: sleepy -> sleepy (no transition)")
should_alert, msg = check_alert_trigger(1, 'sleepy')
print(f"  Alert={should_alert}, Message={msg}")
assert should_alert == False, "Should NOT alert when already sleepy"

# Test 3: Engaged -> Bored (should alert)
print("\n[Test 3] Student 2: engaged -> bored")
should_alert, msg = check_alert_trigger(2, 'engaged')
print(f"  First frame (engaged): Alert={should_alert}")
should_alert, msg = check_alert_trigger(2, 'bored')
print(f"  Second frame (bored): Alert={should_alert}, Message={msg}")
assert should_alert == True, "Should alert on engaged->bored transition"

# Test 4: Neutral -> Yawning (should alert)
print("\n[Test 4] Student 3: neutral -> yawning")
should_alert, msg = check_alert_trigger(3, 'neutral')
print(f"  First frame (neutral): Alert={should_alert}")
should_alert, msg = check_alert_trigger(3, 'yawning')
print(f"  Second frame (yawning): Alert={should_alert}, Message={msg}")
assert should_alert == True, "Should alert on neutral->yawning transition"

# Test 5: Engaged -> Engaged (should NOT alert)
print("\n[Test 5] Student 4: engaged -> engaged (no change)")
should_alert, msg = check_alert_trigger(4, 'engaged')
print(f"  First frame: Alert={should_alert}")
should_alert, msg = check_alert_trigger(4, 'engaged')
print(f"  Second frame: Alert={should_alert}, Message={msg}")
assert should_alert == False, "Should NOT alert when staying engaged"

# Test 6: Bored -> Engaged (should NOT alert - positive change)
print("\n[Test 6] Student 5: bored -> engaged (recovery)")
should_alert, msg = check_alert_trigger(5, 'bored')
print(f"  First frame (bored): Alert={should_alert}")
should_alert, msg = check_alert_trigger(5, 'engaged')
print(f"  Second frame (engaged): Alert={should_alert}, Message={msg}")
assert should_alert == False, "Should NOT alert on recovery to engaged"

# Test 7: Cooldown test (should NOT alert within 10 seconds)
print("\n[Test 7] Student 6: Testing cooldown")
check_alert_trigger(6, 'engaged')
should_alert, msg = check_alert_trigger(6, 'sleepy')
print(f"  First alert (sleepy): Alert={should_alert}, Message={msg}")
assert should_alert == True, "First alert should trigger"

check_alert_trigger(6, 'engaged')
should_alert, msg = check_alert_trigger(6, 'bored')
print(f"  Second alert within cooldown: Alert={should_alert}, Message={msg}")
assert should_alert == False, "Should NOT alert during cooldown period"

print("\n" + "=" * 60)
print("✓ All tests passed! Alert system working correctly.")
print("=" * 60)
print("\nKey Features:")
print("  • Alerts only on state transitions (engaged/neutral -> disengaged)")
print("  • 10-second cooldown per student to prevent spam")
print("  • Specific messages for each disengaged state")
print("  • No alerts for staying in same state or recovering")
