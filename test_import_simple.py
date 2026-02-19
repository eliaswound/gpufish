#!/usr/bin/env python3
"""
Simple import test - run this on Linux to see what happens
Matches the user's actual setup
"""
import sys
import os
import traceback

# Match user's setup: go up two levels to reach planarian_smFISH root
# Try different paths to find where gpufish package should be importable from
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Try going up to find planarian_smFISH root
possible_roots = [
    os.path.abspath(os.path.join(current_dir, "..")),  # One level up
    os.path.abspath(os.path.join(current_dir, "../..")),  # Two levels up
]

project_root = None
for root in possible_roots:
    gpufish_path = os.path.join(root, "gpufish")
    if os.path.exists(gpufish_path) and os.path.exists(os.path.join(gpufish_path, "functions")):
        project_root = root
        break

if project_root is None:
    # Fallback: use current directory's parent
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(project_root)

print("=" * 60)
print("SIMPLE IMPORT TEST (matching your setup)")
print("=" * 60)
print(f"Project root (added to path): {project_root}")
print(f"gpufish should be at: {os.path.join(project_root, 'gpufish')}")
print(f"gpufish exists: {os.path.exists(os.path.join(project_root, 'gpufish'))}")
print(f"Python path includes: {[p for p in sys.path[-3:] if p]}")
print()

# Determine gpufish directory (could be current dir or in project_root)
if os.path.basename(current_dir) == 'gpufish' and os.path.exists(os.path.join(current_dir, 'functions')):
    gpufish_dir = current_dir
else:
    gpufish_dir = os.path.join(project_root, "gpufish")

cursor_dir = os.path.join(gpufish_dir, '.cursor')
os.makedirs(cursor_dir, exist_ok=True)
print(f"✓ Created/verified .cursor directory: {cursor_dir}")

# Test 1: Try importing with different approaches
print("\n[TEST 1] Testing import paths...")

# First, try adding gpufish directory itself to path
if os.path.basename(current_dir) == 'gpufish':
    sys.path.insert(0, current_dir)
    print(f"   → Added current directory to path: {current_dir}")

# Also add project_root/gpufish if it exists
if os.path.exists(os.path.join(project_root, "gpufish")):
    sys.path.insert(0, os.path.join(project_root, "gpufish"))
    print(f"   → Added gpufish from project_root to path")

print("\n[TEST 2] Importing check_image module...")
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

# Test 3: Try importing filter module (this is where the error happens)
print("\n[TEST 3] Importing filter module...")
try:
    from gpufish.functions.core import filter
    print("   ✓ filter module imported")
    
    # Check if check_tiff_dtype is available
    if hasattr(filter, 'check_tiff_dtype'):
        print("   ✓ check_tiff_dtype available in filter module")
        print(f"   → Function: {filter.check_tiff_dtype}")
    else:
        print("   ✗ check_tiff_dtype NOT available in filter module")
        print("   → This is the problem!")
        print(f"   → Available check functions: {[x for x in dir(filter) if 'check' in x.lower()]}")
        print(f"   → All functions: {[x for x in dir(filter) if not x.startswith('_')]}")
        
    # Check log file
    log_path = os.path.join(gpufish_dir, '.cursor', 'debug.log')
    if os.path.exists(log_path):
        print(f"\n   ✓ Log file exists: {log_path}")
        with open(log_path, 'r') as f:
            content = f.read()
            if content:
                lines = content.strip().split('\n')
                print(f"   → Log file has {len(lines)} lines")
                print("   → Last few log entries:")
                for line in lines[-5:]:
                    if line.strip():
                        try:
                            import json
                            log_entry = json.loads(line)
                            print(f"      [{log_entry.get('location', '?')}] {log_entry.get('message', '?')}")
                        except:
                            print(f"      {line[:80]}")
            else:
                print("   → Log file is empty")
    else:
        print(f"\n   ✗ Log file does NOT exist: {log_path}")
        print("   → Instrumentation didn't run or failed silently")
        print("   → Check if filter.py has the instrumentation code")
        
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
