# EngageTrack Test Suite

Comprehensive test suite for the EngageTrack application.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_models.py           # Database model tests
├── test_engagement_logic.py # Engagement detection logic tests
├── test_alert_system.py     # Real-time alert system tests
└── test_routes.py           # Flask route and API tests
```

## Test Categories

### 1. Model Tests (`test_models.py`)
Tests for database models and relationships:
- User model (creation, unique email, password hashing)
- Session model (creation, duration, percentages)
- Detection model (creation, states, privacy)
- Model relationships (user-sessions, session-detections)

**Coverage:**
- User authentication
- Session tracking
- Detection storage
- Data integrity

### 2. Engagement Logic Tests (`test_engagement_logic.py`)
Tests for engagement detection algorithms:
- Engagement state mapping
- IOU (Intersection over Union) calculation
- Student tracking and ID assignment
- Pose feature extraction
- Posture classification
- Signal combination (face + posture)

**Coverage:**
- State classification accuracy
- Tracking algorithms
- Geometric feature extraction
- Multi-signal fusion

### 3. Alert System Tests (`test_alert_system.py`)
Tests for real-time alert system:
- Alert triggering on state transitions
- Cooldown mechanism
- Active alert management
- Alert message formatting
- Edge cases

**Coverage:**
- State transition detection
- Alert spam prevention
- Multi-student tracking
- Message generation

### 4. Route Tests (`test_routes.py`)
Tests for Flask routes and API endpoints:
- Authentication routes (login, signup, logout)
- Protected routes (home, live, reports)
- Detection control (start, stop, toggle)
- Session management (stats, history, details)
- Privacy compliance

**Coverage:**
- API functionality
- Authentication flow
- Session management
- Privacy protection

## Running Tests

### Install Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# Using pytest directly
pytest tests/ -v

# Using test runner script
python run_tests.py

# From project root
python -m pytest tests/
```

### Run Specific Test Files

```bash
# Test models only
pytest tests/test_models.py -v

# Test engagement logic only
pytest tests/test_engagement_logic.py -v

# Test alert system only
pytest tests/test_alert_system.py -v

# Test routes only
pytest tests/test_routes.py -v
```

### Run Specific Test Classes

```bash
# Test User model only
pytest tests/test_models.py::TestUserModel -v

# Test alert triggers only
pytest tests/test_alert_system.py::TestAlertTriggers -v
```

### Run Specific Test Methods

```bash
# Test specific function
pytest tests/test_models.py::TestUserModel::test_create_user -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/ --cov=app --cov-report=html --cov-report=term

# View HTML report
# Open htmlcov/index.html in browser
```

### Run with Markers

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"
```

## Test Output

### Successful Run
```
tests/test_models.py::TestUserModel::test_create_user PASSED
tests/test_models.py::TestUserModel::test_user_unique_email PASSED
tests/test_engagement_logic.py::TestEngagementMapping::test_engaged_states PASSED
...
======================== 50 passed in 2.34s ========================
```

### Failed Test
```
tests/test_models.py::TestUserModel::test_create_user FAILED

FAILED tests/test_models.py::TestUserModel::test_create_user
AssertionError: assert None is not None
```

## Writing New Tests

### Test Structure

```python
import pytest

class TestFeatureName:
    """Test description"""
    
    def setup_method(self):
        """Setup before each test"""
        pass
    
    def test_specific_behavior(self):
        """Test specific behavior"""
        # Arrange
        input_data = "test"
        
        # Act
        result = function_to_test(input_data)
        
        # Assert
        assert result == expected_output
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """Test using fixture"""
    assert sample_data["key"] == "value"
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("engaged", "engaged"),
    ("bored", "bored"),
    ("sleepy", "sleepy"),
])
def test_multiple_cases(input, expected):
    """Test multiple cases"""
    assert get_engagement_state(input) == expected
```

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Clear Test Names
```python
# Good
def test_user_creation_with_valid_email():
    pass

# Bad
def test_user():
    pass
```

### 3. Arrange-Act-Assert Pattern
```python
def test_example():
    # Arrange - setup test data
    user = User(email="test@example.com")
    
    # Act - perform action
    result = user.validate()
    
    # Assert - verify result
    assert result is True
```

### 4. Test Edge Cases
- Empty inputs
- Null values
- Boundary conditions
- Error conditions

### 5. Use Descriptive Assertions
```python
# Good
assert len(results) == 3, f"Expected 3 results, got {len(results)}"

# Bad
assert len(results) == 3
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ --cov=app
```

## Troubleshooting

### Import Errors
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Database Errors
- Tests use in-memory SQLite database
- Each test gets fresh database
- No need to clean up manually

### Fixture Not Found
- Check fixture is in conftest.py or same file
- Verify fixture name matches parameter name

### Test Discovery Issues
```bash
# Verify pytest can find tests
pytest --collect-only tests/
```

## Coverage Goals

Target coverage by module:
- Models: 95%+
- Engagement Logic: 90%+
- Alert System: 95%+
- Routes: 85%+

Overall target: 90%+

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain coverage above 90%
4. Update this README if needed

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Flask Testing](https://flask.palletsprojects.com/en/2.0.x/testing/)
- [SQLAlchemy Testing](https://docs.sqlalchemy.org/en/14/orm/session_transaction.html)
