#!/usr/bin/env python3
"""
Test script to reproduce the actual import failure.
Run this on Linux: python test_import_runtime.py
"""
import sys
import os
import traceback

# Simulate the user's setup
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.insert(0, project_root)

print("=" * 60)
print("TESTING ACTUAL IMPORT PATH")
print("=" * 60)
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")
print()

# Test 1: Try importing check_image module directly
print("[TEST 1] Importing check_image module directly...")
try:
    from gpufish.functions.checks import check_image
    print("   ✓ Module imported")
    print(f"   → Module: {check_image}")
    print(f"   → Has check_tiff_dtype: {hasattr(check_image, 'check_tiff_dtype')}")
    if hasattr(check_image, 'check_tiff_dtype'):
        print(f"   → Function: {check_image.check_tiff_dtype}")
        print(f"   → Callable: {callable(check_image.check_tiff_dtype)}")
    else:
        print("   ✗ check_tiff_dtype NOT in module!")
        print(f"   → Available: {[x for x in dir(check_image) if not x.startswith('_')]}")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print()

# Test 2: Try importing check_tiff_dtype directly
print("[TEST 2] Importing check_tiff_dtype directly...")
try:
    from gpufish.functions.checks.check_image import check_tiff_dtype
    print("   ✓ Imported successfully")
    print(f"   → Function: {check_tiff_dtype}")
    print(f"   → Callable: {callable(check_tiff_dtype)}")
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print()

# Test 3: Try importing filter module (this is where the error happens)
print("[TEST 3] Importing filter module (where error occurs)...")
try:
    from gpufish.functions.core import filter
    print("   ✓ filter module imported")
    print(f"   → Module: {filter}")
    
    # Check if check_tiff_dtype is available in filter module
    if hasattr(filter, 'check_tiff_dtype'):
        print("   ✓ check_tiff_dtype available in filter module")
    else:
        print("   ✗ check_tiff_dtype NOT available in filter module")
        print("   → This is the problem!")
        
    # Check what's actually imported
    print(f"   → Module file: {filter.__file__}")
    print(f"   → Module dict keys: {[k for k in dir(filter) if not k.startswith('_')]}")
    
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print()

# Test 4: Try calling log_filter to see the actual error
print("[TEST 4] Testing log_filter function call...")
try:
    from gpufish.functions.core.filter import log_filter
    print("   ✓ log_filter imported")
    # Don't actually call it, just check if it exists
    print(f"   → Function: {log_filter}")
except NameError as e:
    print(f"   ✗ NameError: {e}")
    print("   → This matches your error!")
    traceback.print_exc()
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test complete!")
print("=" * 60)
