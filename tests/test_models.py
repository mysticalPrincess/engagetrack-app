"""
Test database models
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, db, User, Session, Detection
from werkzeug.security import check_password_hash


@pytest.fixture
def test_app():
    """Create test application with in-memory database"""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(test_app):
    """Create test client"""
    return test_app.test_client()


class TestUserModel:
    """Test User model"""
    
    def test_create_user(self, test_app):
        """Test creating a user"""
        with test_app.app_context():
            user = User(
                email='test@example.com',
                password_hash='hashed_password'
            )
            db.session.add(user)
            db.session.commit()
            
            assert user.id is not None
            assert user.email == 'test@example.com'
            assert user.created_at is not None
    
    def test_user_unique_email(self, test_app):
        """Test that email must be unique"""
        with test_app.app_context():
            user1 = User(email='test@example.com', password_hash='hash1')
            db.session.add(user1)
            db.session.commit()
            
            user2 = User(email='test@example.com', password_hash='hash2')
            db.session.add(user2)
            
            with pytest.raises(Exception):  # Should raise IntegrityError
                db.session.commit()
    
    def test_password_hashing(self, test_app):
        """Test password hashing"""
        with test_app.app_context():
            from werkzeug.security import generate_password_hash
            
            password = 'secure_password123'
            hashed = generate_password_hash(password)
            
            user = User(email='test@example.com', password_hash=hashed)
            db.session.add(user)
            db.session.commit()
            
            # Verify password can be checked
            assert check_password_hash(user.password_hash, password)
            assert not check_password_hash(user.password_hash, 'wrong_password')


class TestSessionModel:
    """Test Session model"""
    
    def test_create_session(self, test_app):
        """Test creating a session"""
        with test_app.app_context():
            user = User(email='test@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
            
            session = Session(
                user_id=user.id,
                total_students=5,
                engaged_percentage=75.0,
                neutral_percentage=15.0,
                bored_percentage=10.0
            )
            db.session.add(session)
            db.session.commit()
            
            assert session.id is not None
            assert session.user_id == user.id
            assert session.start_time is not None
            assert session.total_students == 5
    
    def test_session_duration_calculation(self, test_app):
        """Test session duration calculation"""
        with test_app.app_context():
            user = User(email='test@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
            
            session = Session(user_id=user.id)
            session.start_time = datetime.utcnow() - timedelta(minutes=30)
            session.end_time = datetime.utcnow()
            session.duration = int((session.end_time - session.start_time).total_seconds())
            
            db.session.add(session)
            db.session.commit()
            
            assert session.duration >= 1800  # At least 30 minutes
            assert session.duration <= 1810  # Allow small variance
    
    def test_session_percentages(self, test_app):
        """Test engagement percentages"""
        with test_app.app_context():
            user = User(email='test@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
            
            session = Session(
                user_id=user.id,
                engaged_percentage=60.5,
                neutral_percentage=25.3,
                bored_percentage=14.2
            )
            db.session.add(session)
            db.session.commit()
            
            # Percentages should sum to ~100
            total = session.engaged_percentage + session.neutral_percentage + session.bored_percentage
            assert 99.0 <= total <= 101.0


class TestDetectionModel:
    """Test Detection model"""
    
    def test_create_detection(self, test_app):
        """Test creating a detection"""
        with test_app.app_context():
            user = User(email='test@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
            
            session = Session(user_id=user.id)
            db.session.add(session)
            db.session.commit()
            
            detection = Detection(
                session_id=session.id,
                student_id=1,
                engagement_state='engaged',
                confidence=0.95,
                image_path=None
            )
            db.session.add(detection)
            db.session.commit()
            
            assert detection.id is not None
            assert detection.session_id == session.id
            assert detection.student_id == 1
            assert detection.engagement_state == 'engaged'
            assert detection.confidence == 0.95
            assert detection.timestamp is not None
    
    def test_detection_states(self, test_app):
        """Test different engagement states"""
        with test_app.app_context():
            user = User(email='test@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
            
            session = Session(user_id=user.id)
            db.session.add(session)
            db.session.commit()
            
            states = ['engaged', 'neutral', 'bored', 'sleepy', 'yawning', 'frustrated', 'confused']
            
            for i, state in enumerate(states):
                detection = Detection(
                    session_id=session.id,
                    student_id=i + 1,
                    engagement_state=state,
                    confidence=0.8
                )
                db.session.add(detection)
            
            db.session.commit()
            
            # Verify all states saved
            detections = Detection.query.filter_by(session_id=session.id).all()
            assert len(detections) == len(states)
            
            saved_states = [d.engagement_state for d in detections]
            for state in states:
                assert state in saved_states
    
    def test_detection_no_image_path(self, test_app):
        """Test that image_path is None (privacy)"""
        with test_app.app_context():
            user = User(email='test@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
            
            session = Session(user_id=user.id)
            db.session.add(session)
            db.session.commit()
            
            detection = Detection(
                session_id=session.id,
                student_id=1,
                engagement_state='engaged',
                confidence=0.9,
                image_path=None
            )
            db.session.add(detection)
            db.session.commit()
            
            # Verify no image path saved
            assert detection.image_path is None


class TestModelRelationships:
    """Test relationships between models"""
    
    def test_user_sessions_relationship(self, test_app):
        """Test user can have multiple sessions"""
        with test_app.app_context():
            user = User(email='test@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
            
            # Create multiple sessions
            for i in range(3):
                session = Session(user_id=user.id, total_students=i + 1)
                db.session.add(session)
            db.session.commit()
            
            # Query sessions for user
            sessions = Session.query.filter_by(user_id=user.id).all()
            assert len(sessions) == 3
    
    def test_session_detections_relationship(self, test_app):
        """Test session can have multiple detections"""
        with test_app.app_context():
            user = User(email='test@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
            
            session = Session(user_id=user.id)
            db.session.add(session)
            db.session.commit()
            
            # Create multiple detections
            for i in range(10):
                detection = Detection(
                    session_id=session.id,
                    student_id=i % 3 + 1,  # 3 students
                    engagement_state='engaged',
                    confidence=0.9
                )
                db.session.add(detection)
            db.session.commit()
            
            # Query detections for session
            detections = Detection.query.filter_by(session_id=session.id).all()
            assert len(detections) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
