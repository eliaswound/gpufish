#!/usr/bin/env python3
"""
Simple test to verify logging works
Run this on Linux: python test_logging.py
"""
import sys
import os

# Add project root
project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.insert(0, project_root)

print("Testing import and logging...")
print(f"Project root: {project_root}")
print(f".cursor path: {os.path.join(project_root, '.cursor', 'debug.log')}")

try:
    # Try importing filter module
    print("\n[1] Importing filter module...")
    from gpufish.functions.core import filter
    print("   ✓ Import successful")
    
    # Check if check_tiff_dtype is available
    print("\n[2] Checking if check_tiff_dtype is available...")
    if hasattr(filter, 'check_tiff_dtype'):
        print("   ✓ check_tiff_dtype available")
        print(f"   → Function: {filter.check_tiff_dtype}")
    else:
        print("   ✗ check_tiff_dtype NOT available")
        print(f"   → Available: {[x for x in dir(filter) if 'check' in x.lower()]}")
    
    # Check debug log
    log_path = os.path.join(project_root, '.cursor', 'debug.log')
    print(f"\n[3] Checking debug log: {log_path}")
    if os.path.exists(log_path):
        print("   ✓ Log file exists")
        with open(log_path, 'r') as f:
            lines = f.readlines()
            print(f"   → {len(lines)} log entries")
            for i, line in enumerate(lines[-5:], 1):  # Show last 5 lines
                print(f"   [{i}] {line.strip()[:100]}")
    else:
        print("   ✗ Log file does NOT exist")
        print("   → This means the instrumentation didn't run")
        print("   → Check if filter.py has the instrumentation code")
        
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
