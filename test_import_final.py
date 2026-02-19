#!/usr/bin/env python3
"""
Final import test - matches your exact code setup
Run from gpufish directory: python test_import_final.py
"""
import sys
import os
import traceback

# EXACTLY match your code setup
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(project_root)

print("=" * 60)
print("FINAL IMPORT TEST (exact match of your code)")
print("=" * 60)
print(f"Current directory: {os.getcwd()}")
print(f"Project root added to path: {project_root}")
print(f"gpufish should be at: {os.path.join(project_root, 'gpufish')}")
print(f"gpufish exists: {os.path.exists(os.path.join(project_root, 'gpufish'))}")
print()

# Ensure .cursor directory exists
cursor_dir = os.path.join(os.getcwd(), '.cursor')
os.makedirs(cursor_dir, exist_ok=True)
log_path = os.path.join(cursor_dir, 'debug.log')
print(f"Log file will be at: {log_path}")
print()

# Test 1: Import check_image module
print("[TEST 1] Importing check_image module...")
try:
    from gpufish.functions.checks import check_image
    print("   ✓ Module imported successfully")
    print(f"   → Module file: {getattr(check_image, '__file__', 'unknown')}")
    print(f"   → Has check_tiff_dtype: {hasattr(check_image, 'check_tiff_dtype')}")
    
    if hasattr(check_image, 'check_tiff_dtype'):
        print(f"   → Function exists: {check_image.check_tiff_dtype}")
        print(f"   → Callable: {callable(check_image.check_tiff_dtype)}")
    else:
        print("   ✗ check_tiff_dtype NOT FOUND in module!")
        all_funcs = [x for x in dir(check_image) if not x.startswith('_') and callable(getattr(check_image, x, None))]
        print(f"   → Available functions: {all_funcs}")
        
except Exception as e:
    print(f"   ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import filter module (where the error occurs)
print("\n[TEST 2] Importing filter module...")
try:
    from gpufish.functions.core import filter
    print("   ✓ filter module imported")
    
    # Check if check_tiff_dtype is available
    if hasattr(filter, 'check_tiff_dtype'):
        print("   ✓ check_tiff_dtype IS available in filter module!")
        print(f"   → Function: {filter.check_tiff_dtype}")
    else:
        print("   ✗ check_tiff_dtype NOT available in filter module")
        print("   → This is the problem!")
        check_funcs = [x for x in dir(filter) if 'check' in x.lower()]
        print(f"   → Available check functions: {check_funcs}")
        all_funcs = [x for x in dir(filter) if not x.startswith('_')]
        print(f"   → All functions: {all_funcs}")
        
    # Check log file
    print(f"\n[TEST 3] Checking debug log: {log_path}")
    if os.path.exists(log_path):
        print("   ✓ Log file exists")
        with open(log_path, 'r') as f:
            content = f.read().strip()
            if content:
                lines = content.split('\n')
                print(f"   → {len(lines)} log entries found")
                print("   → Last 3 entries:")
                for i, line in enumerate(lines[-3:], 1):
                    if line.strip():
                        try:
                            import json
                            entry = json.loads(line)
                            loc = entry.get('location', '?')
                            msg = entry.get('message', '?')
                            print(f"      [{i}] {loc}: {msg}")
                        except:
                            print(f"      [{i}] {line[:70]}")
            else:
                print("   → Log file is empty (instrumentation didn't run)")
    else:
        print("   ✗ Log file does NOT exist")
        print("   → Instrumentation code didn't execute")
        
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
