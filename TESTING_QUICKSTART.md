# Testing Quick Start Guide

Get started with testing EngageTrack in 5 minutes!

## Step 1: Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

This installs:
- pytest (testing framework)
- pytest-cov (coverage reporting)
- pytest-flask (Flask testing utilities)

## Step 2: Run All Tests

```bash
python run_tests.py
```

Expected output:
```
============================================================
EngageTrack Test Suite
============================================================
tests/test_models.py::TestUserModel::test_create_user PASSED
tests/test_models.py::TestUserModel::test_user_unique_email PASSED
...
======================== 78 passed in 3.45s ========================

============================================================
✓ All tests passed!
============================================================
```

## Step 3: Run Specific Test Suites

### Test Database Models
```bash
pytest tests/test_models.py -v
```

### Test Engagement Logic
```bash
pytest tests/test_engagement_logic.py -v
```

### Test Alert System
```bash
pytest tests/test_alert_system.py -v
```

### Test API Routes
```bash
pytest tests/test_routes.py -v
```

## Step 4: Check Code Coverage

```bash
pytest tests/ --cov=app --cov-report=term
```

Output shows coverage percentage:
```
Name                    Stmts   Miss  Cover
-------------------------------------------
app.py                    450     45    90%
-------------------------------------------
TOTAL                     450     45    90%
```

## Step 5: Generate HTML Coverage Report

```bash
pytest tests/ --cov=app --cov-report=html
```

Then open `htmlcov/index.html` in your browser to see detailed coverage.

## Common Commands

### Run Single Test
```bash
pytest tests/test_models.py::TestUserModel::test_create_user -v
```

### Run Tests Matching Pattern
```bash
pytest tests/ -k "alert" -v
```

### Show Print Statements
```bash
pytest tests/ -v -s
```

### Stop on First Failure
```bash
pytest tests/ -x
```

### Run Last Failed Tests
```bash
pytest tests/ --lf
```

## Understanding Test Output

### ✅ Passed Test
```
tests/test_models.py::TestUserModel::test_create_user PASSED
```

### ❌ Failed Test
```
tests/test_models.py::TestUserModel::test_create_user FAILED

AssertionError: assert None is not None
```

### ⚠️ Skipped Test
```
tests/test_models.py::TestUserModel::test_slow_operation SKIPPED
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'app'"
```bash
# Run from project root directory
cd /path/to/engagetrack-app
pytest tests/
```

### "No tests collected"
```bash
# Verify test files exist
ls tests/test_*.py

# Check pytest can find them
pytest --collect-only tests/
```

### Database Errors
Tests use in-memory SQLite - no setup needed!

### Import Errors
```bash
# Ensure you're in the project root
pwd
# Should show: /path/to/engagetrack-app

# Run tests
pytest tests/
```

## What's Being Tested?

### ✅ Database Models (18 tests)
- User creation and authentication
- Session tracking
- Detection storage
- Privacy compliance

### ✅ Engagement Logic (20 tests)
- State classification
- Student tracking
- Pose analysis
- Signal fusion

### ✅ Alert System (20 tests)
- State transitions
- Cooldown mechanism
- Multi-student tracking
- Message formatting

### ✅ API Routes (20 tests)
- Authentication
- Session management
- Detection control
- Data retrieval

## Next Steps

1. **Run tests regularly** - Before committing code
2. **Add new tests** - When adding features
3. **Check coverage** - Aim for 90%+
4. **Read docs** - See `tests/README.md` for details

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python run_tests.py` | Run all tests |
| `pytest tests/ -v` | Verbose output |
| `pytest tests/ --cov=app` | With coverage |
| `pytest tests/ -k "alert"` | Run specific tests |
| `pytest tests/ -x` | Stop on first fail |
| `pytest tests/ --lf` | Run last failed |

## Help

For detailed documentation:
- `tests/README.md` - Complete testing guide
- `TEST_SUITE_SUMMARY.md` - Test suite overview
- [Pytest Docs](https://docs.pytest.org/) - Official documentation

## Success!

If you see "✓ All tests passed!" - you're good to go! 🎉

The test suite ensures:
- ✅ All features work correctly
- ✅ Privacy is maintained
- ✅ No regressions occur
- ✅ Code quality is high
