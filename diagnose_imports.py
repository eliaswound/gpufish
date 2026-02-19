#!/usr/bin/env python3
"""
Diagnostic script to check why check_tiff_dtype import fails.
Run this on your Linux machine: python diagnose_imports.py
"""
import sys
import os

# Add project root to path (adjust if needed)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

print("=" * 60)
print("DIAGNOSTIC: Checking check_tiff_dtype import issue")
print("=" * 60)

# Check 1: Does check_image.py exist?
check_image_path = os.path.join(project_root, "functions", "checks", "check_image.py")
print(f"\n[1] Checking if check_image.py exists: {check_image_path}")
if os.path.exists(check_image_path):
    print("   ✓ File exists")
else:
    print("   ✗ File NOT FOUND!")
    sys.exit(1)

# Check 2: Does check_tiff_dtype function exist in check_image.py?
print(f"\n[2] Checking if check_tiff_dtype function exists in check_image.py")
with open(check_image_path, 'r') as f:
    content = f.read()
    if 'def check_tiff_dtype' in content:
        print("   ✓ Function definition found")
        # Find the line number
        for i, line in enumerate(content.split('\n'), 1):
            if 'def check_tiff_dtype' in line:
                print(f"   → Found at line {i}: {line.strip()}")
    else:
        print("   ✗ Function definition NOT FOUND!")
        print("   → This is the problem! check_tiff_dtype is missing from check_image.py")

# Check 3: Does filter.py import check_tiff_dtype?
filter_path = os.path.join(project_root, "functions", "core", "filter.py")
print(f"\n[3] Checking if filter.py imports check_tiff_dtype: {filter_path}")
if os.path.exists(filter_path):
    print("   ✓ File exists")
    with open(filter_path, 'r') as f:
        content = f.read()
        if 'check_tiff_dtype' in content:
            print("   ✓ check_tiff_dtype found in file")
            # Check if it's in an import statement
            import_lines = [line for line in content.split('\n') if 'from' in line and 'check_image' in line]
            for line in import_lines:
                if 'check_tiff_dtype' in line:
                    print(f"   → Import found: {line.strip()}")
                    break
            else:
                print("   ✗ check_tiff_dtype found but NOT in import statement!")
        else:
            print("   ✗ check_tiff_dtype NOT FOUND in filter.py!")
            print("   → This is the problem! Import statement is missing")
else:
    print("   ✗ filter.py NOT FOUND!")

# Check 4: Try to import and see what happens
print(f"\n[4] Attempting to import check_tiff_dtype")
try:
    from gpufish.functions.checks.check_image import check_tiff_dtype
    print("   ✓ Import successful!")
    print(f"   → Function: {check_tiff_dtype}")
    print(f"   → Callable: {callable(check_tiff_dtype)}")
except ImportError as e:
    print(f"   ✗ Import FAILED!")
    print(f"   → Error: {e}")
except NameError as e:
    print(f"   ✗ NameError!")
    print(f"   → Error: {e}")
except Exception as e:
    print(f"   ✗ Unexpected error!")
    print(f"   → Error: {type(e).__name__}: {e}")

# Check 5: Try to import filter module
print(f"\n[5] Attempting to import filter module")
try:
    from gpufish.functions.core import filter
    print("   ✓ filter module imported successfully!")
    if hasattr(filter, 'check_tiff_dtype'):
        print("   ✓ check_tiff_dtype available in filter module")
    else:
        print("   ✗ check_tiff_dtype NOT available in filter module")
        print("   → Available attributes:", [x for x in dir(filter) if not x.startswith('_')])
except ImportError as e:
    print(f"   ✗ Import FAILED!")
    print(f"   → Error: {e}")
except NameError as e:
    print(f"   ✗ NameError!")
    print(f"   → Error: {e}")
except Exception as e:
    print(f"   ✗ Unexpected error!")
    print(f"   → Error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Diagnostic complete!")
print("=" * 60)
