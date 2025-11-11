"""
Verify that no student images are being saved
Checks code for any remaining snapshot functionality
"""

import os
import re

print("=" * 60)
print("Privacy Verification Script")
print("=" * 60)

# Files to check
files_to_check = ['app.py', 'templates/reports.html']

# Patterns that should NOT exist
forbidden_patterns = [
    (r'save_student_snapshot', 'save_student_snapshot function'),
    (r'cv2\.imwrite.*student', 'Image writing for students'),
    (r'student_snapshots\s*=\s*\{', 'student_snapshots dictionary'),
    (r'session_snapshots', 'session_snapshots directory'),
    (r'\.snapshot-', 'Snapshot CSS classes'),
    (r'student_images', 'student_images variable'),
]

# Patterns that SHOULD exist (privacy-safe)
required_patterns = [
    (r"image_path['\"]?\s*:\s*None", 'image_path set to None'),
]

issues_found = []
checks_passed = []

print("\n[1] Checking for forbidden patterns...")
print("-" * 60)

for file_path in files_to_check:
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for pattern, description in forbidden_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues_found.append(f"✗ {file_path}: Found {description}")
            print(f"✗ {file_path}: Found {description}")
            for match in matches[:3]:  # Show first 3 matches
                print(f"    → {match}")
        else:
            checks_passed.append(f"✓ {file_path}: No {description}")

print("\n[2] Checking for required privacy patterns...")
print("-" * 60)

for file_path in ['app.py']:
    if not os.path.exists(file_path):
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for pattern, description in required_patterns:
        if re.search(pattern, content):
            checks_passed.append(f"✓ {file_path}: {description} found")
            print(f"✓ {file_path}: {description} found")
        else:
            issues_found.append(f"✗ {file_path}: Missing {description}")
            print(f"✗ {file_path}: Missing {description}")

print("\n[3] Checking for snapshot directories...")
print("-" * 60)

snapshot_dirs = [
    'static/session_snapshots',
    'session_snapshots',
    'snapshots'
]

for dir_path in snapshot_dirs:
    if os.path.exists(dir_path):
        file_count = sum(len(files) for _, _, files in os.walk(dir_path))
        print(f"⚠ Directory exists: {dir_path} ({file_count} files)")
        print(f"   → Can be safely deleted (not used by app)")
    else:
        checks_passed.append(f"✓ No directory: {dir_path}")
        print(f"✓ No directory: {dir_path}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if issues_found:
    print(f"\n⚠ {len(issues_found)} issue(s) found:\n")
    for issue in issues_found:
        print(f"  {issue}")
    print("\n❌ Privacy verification FAILED")
else:
    print(f"\n✓ All {len(checks_passed)} checks passed")
    print("\n✅ Privacy verification PASSED")
    print("\nThe application does NOT save student images.")

print("=" * 60)
