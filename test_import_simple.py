#!/usr/bin/env python3
"""
Simple import test - run this on Linux to see what happens
"""
import sys
import os
import traceback

project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.insert(0, project_root)

print("=" * 60)
print("SIMPLE IMPORT TEST")
print("=" * 60)
print(f"Working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print()

# Ensure .cursor directory exists
cursor_dir = os.path.join(project_root, '.cursor')
os.makedirs(cursor_dir, exist_ok=True)
print(f"✓ Created/verified .cursor directory: {cursor_dir}")

# Test 1: Import check_image module directly
print("\n[TEST 1] Importing check_image module...")
try:
    from gpufish.functions.checks import check_image
    print("   ✓ Module imported")
    print(f"   → Module file: {getattr(check_image, '__file__', 'unknown')}")
    print(f"   → Has check_tiff_dtype: {hasattr(check_image, 'check_tiff_dtype')}")
    
    if hasattr(check_image, 'check_tiff_dtype'):
        print(f"   → Function: {check_image.check_tiff_dtype}")
        print(f"   → Callable: {callable(check_image.check_tiff_dtype)}")
    else:
        print("   ✗ check_tiff_dtype NOT FOUND!")
        print(f"   → Available functions: {[x for x in dir(check_image) if not x.startswith('_') and callable(getattr(check_image, x, None))]}")
        
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Try importing filter module (this is where the error happens)
print("\n[TEST 2] Importing filter module...")
try:
    from gpufish.functions.core import filter
    print("   ✓ filter module imported")
    
    # Check if check_tiff_dtype is available
    if hasattr(filter, 'check_tiff_dtype'):
        print("   ✓ check_tiff_dtype available in filter module")
    else:
        print("   ✗ check_tiff_dtype NOT available in filter module")
        print("   → This is the problem!")
        print(f"   → Available: {[x for x in dir(filter) if 'check' in x.lower()]}")
        
    # Check log file
    log_path = os.path.join(project_root, '.cursor', 'debug.log')
    if os.path.exists(log_path):
        print(f"\n   ✓ Log file exists: {log_path}")
        with open(log_path, 'r') as f:
            content = f.read()
            if content:
                print(f"   → Log file has {len(content.split(chr(10)))} lines")
                print("   → Last few lines:")
                for line in content.split(chr(10))[-3:]:
                    if line.strip():
                        print(f"      {line[:80]}")
            else:
                print("   → Log file is empty")
    else:
        print(f"\n   ✗ Log file does NOT exist: {log_path}")
        print("   → Instrumentation didn't run or failed silently")
        
except NameError as e:
    print(f"   ✗ NameError: {e}")
    print("   → This matches your original error!")
    traceback.print_exc()
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
