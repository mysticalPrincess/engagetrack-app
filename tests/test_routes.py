"""
Test Flask routes and API endpoints
"""

import pytest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, db, User, Session, Detection
from werkzeug.security import generate_password_hash


@pytest.fixture
def test_app():
    """Create test application"""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SECRET_KEY'] = 'test-secret-key'
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(test_app):
    """Create test client"""
    return test_app.test_client()


@pytest.fixture
def authenticated_client(test_app, client):
    """Create authenticated test client"""
    with test_app.app_context():
        # Create test user
        user = User(
            email='test@example.com',
            password_hash=generate_password_hash('password123')
        )
        db.session.add(user)
        db.session.commit()
        user_id = user.id
    
    # Login
    with client.session_transaction() as sess:
        sess['user_id'] = user_id
    
    return client


class TestAuthRoutes:
    """Test authentication routes"""
    
    def test_login_page_loads(self, client):
        """Test login page loads"""
        response = client.get('/login')
        assert response.status_code == 200
    
    def test_signup_creates_user(self, test_app, client):
        """Test signup creates new user"""
        response = client.post('/signup',
            data=json.dumps({
                'email': 'newuser@example.com',
                'password': 'password123'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        
        # Verify user created
        with test_app.app_context():
            user = User.query.filter_by(email='newuser@example.com').first()
            assert user is not None
    
    def test_signup_duplicate_email(self, test_app, client):
        """Test signup with duplicate email fails"""
        with test_app.app_context():
            user = User(email='existing@example.com', password_hash='hash')
            db.session.add(user)
            db.session.commit()
        
        response = client.post('/signup',
            data=json.dumps({
                'email': 'existing@example.com',
                'password': 'password123'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
    
    def test_login_success(self, test_app, client):
        """Test successful login"""
        with test_app.app_context():
            user = User(
                email='test@example.com',
                password_hash=generate_password_hash('password123')
            )
            db.session.add(user)
            db.session.commit()
        
        response = client.post('/login',
            data=json.dumps({
                'email': 'test@example.com',
                'password': 'password123'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
    
    def test_login_invalid_credentials(self, test_app, client):
        """Test login with invalid credentials"""
        with test_app.app_context():
            user = User(
                email='test@example.com',
                password_hash=generate_password_hash('password123')
            )
            db.session.add(user)
            db.session.commit()
        
        response = client.post('/login',
            data=json.dumps({
                'email': 'test@example.com',
                'password': 'wrongpassword'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 401
    
    def test_logout(self, authenticated_client):
        """Test logout"""
        response = authenticated_client.get('/logout')
        assert response.status_code == 302  # Redirect


class TestProtectedRoutes:
    """Test routes that require authentication"""
    
    def test_home_requires_auth(self, client):
        """Test home page requires authentication"""
        response = client.get('/')
        assert response.status_code == 302  # Redirect to login
    
    def test_live_requires_auth(self, client):
        """Test live page requires authentication"""
        response = client.get('/live')
        assert response.status_code == 302
    
    def test_reports_requires_auth(self, client):
        """Test reports page requires authentication"""
        response = client.get('/reports')
        assert response.status_code == 302
    
    def test_authenticated_home_access(self, authenticated_client):
        """Test authenticated user can access home"""
        response = authenticated_client.get('/')
        assert response.status_code == 200
    
    def test_authenticated_live_access(self, authenticated_client):
        """Test authenticated user can access live"""
        response = authenticated_client.get('/live')
        assert response.status_code == 200
    
    def test_authenticated_reports_access(self, authenticated_client):
        """Test authenticated user can access reports"""
        response = authenticated_client.get('/reports')
        assert response.status_code == 200


class TestDetectionRoutes:
    """Test detection control routes"""
    
    def test_start_detection_requires_auth(self, client):
        """Test start detection requires authentication"""
        response = client.post('/start_detection')
        assert response.status_code == 401
    
    def test_start_detection_creates_session(self, test_app, authenticated_client):
        """Test start detection creates session"""
        response = authenticated_client.post('/start_detection')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'session_id' in data
        
        # Verify session created
        with test_app.app_context():
            session = Session.query.get(data['session_id'])
            assert session is not None
    
    def test_stop_detection(self, test_app, authenticated_client):
        """Test stop detection"""
        # Start detection first
        authenticated_client.post('/start_detection')
        
        # Stop detection
        response = authenticated_client.post('/stop_detection')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
    
    def test_toggle_pose(self, authenticated_client):
        """Test toggle pose detection"""
        response = authenticated_client.post('/toggle_pose')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'pose_active' in data


class TestSessionRoutes:
    """Test session-related routes"""
    
    def test_session_stats_no_active_session(self, authenticated_client):
        """Test session stats with no active session"""
        response = authenticated_client.get('/session_stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['active'] is False
    
    def test_session_stats_active_session(self, test_app, authenticated_client):
        """Test session stats with active session"""
        # Start detection
        start_response = authenticated_client.post('/start_detection')
        session_id = json.loads(start_response.data)['session_id']
        
        # Get stats
        response = authenticated_client.get('/session_stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['active'] is True
        assert 'duration' in data
        assert 'students' in data
    
    def test_past_sessions_requires_auth(self, client):
        """Test past sessions requires authentication"""
        response = client.get('/past_sessions')
        assert response.status_code == 401
    
    def test_past_sessions_empty(self, authenticated_client):
        """Test past sessions when none exist"""
        response = authenticated_client.get('/past_sessions')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'sessions' in data
        assert len(data['sessions']) == 0
    
    def test_past_sessions_with_data(self, test_app, authenticated_client):
        """Test past sessions with completed sessions"""
        with test_app.app_context():
            # Get user
            user = User.query.filter_by(email='test@example.com').first()
            
            # Create completed session
            from datetime import datetime, timedelta
            session = Session(
                user_id=user.id,
                start_time=datetime.utcnow() - timedelta(hours=1),
                end_time=datetime.utcnow(),
                duration=3600,
                total_students=5,
                engaged_percentage=75.0,
                neutral_percentage=15.0,
                bored_percentage=10.0
            )
            db.session.add(session)
            db.session.commit()
        
        response = authenticated_client.get('/past_sessions')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['sessions']) == 1
        assert data['sessions'][0]['students'] == 5
    
    def test_session_details(self, test_app, authenticated_client):
        """Test session details endpoint"""
        with test_app.app_context():
            # Get user
            user = User.query.filter_by(email='test@example.com').first()
            
            # Create session with detections
            from datetime import datetime
            session = Session(
                user_id=user.id,
                end_time=datetime.utcnow(),
                duration=1800,
                total_students=3,
                engaged_percentage=70.0,
                neutral_percentage=20.0,
                bored_percentage=10.0
            )
            db.session.add(session)
            db.session.commit()
            
            # Add detections
            for i in range(5):
                detection = Detection(
                    session_id=session.id,
                    student_id=1,
                    engagement_state='engaged',
                    confidence=0.9
                )
                db.session.add(detection)
            db.session.commit()
            
            session_id = session.id
        
        response = authenticated_client.get(f'/session_details/{session_id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'session' in data
        assert 'student_data' in data
        assert data['session']['students'] == 3


class TestPrivacy:
    """Test privacy-related functionality"""
    
    def test_no_image_paths_in_response(self, test_app, authenticated_client):
        """Test that image paths are not returned"""
        with test_app.app_context():
            user = User.query.filter_by(email='test@example.com').first()
            
            session = Session(user_id=user.id, end_time=datetime.utcnow())
            db.session.add(session)
            db.session.commit()
            
            detection = Detection(
                session_id=session.id,
                student_id=1,
                engagement_state='engaged',
                confidence=0.9,
                image_path=None  # Should always be None
            )
            db.session.add(detection)
            db.session.commit()
            
            session_id = session.id
        
        response = authenticated_client.get(f'/session_details/{session_id}')
        data = json.loads(response.data)
        
        # Verify no student_images in response
        assert 'student_images' not in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
