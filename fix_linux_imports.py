#!/usr/bin/env python3
"""
Comprehensive fix script for Linux machine.
Run: python fix_linux_imports.py
"""
import os
import sys
import shutil

def fix_filter_imports():
    """Fix imports in filter.py"""
    filter_path = "functions/core/filter.py"
    
    if not os.path.exists(filter_path):
        print(f"Error: {filter_path} not found!")
        return False
    
    print(f"Reading {filter_path}...")
    with open(filter_path, 'r') as f:
        lines = f.readlines()
    
    # Find the import line
    import_line_idx = None
    for i, line in enumerate(lines):
        if 'from ..checks.check_image import' in line:
            import_line_idx = i
            break
    
    if import_line_idx is None:
        print("Error: Could not find import line!")
        return False
    
    import_line = lines[import_line_idx]
    print(f"Current import line ({import_line_idx+1}): {import_line.strip()}")
    
    # Check if check_tiff_dtype is in the import
    if 'check_tiff_dtype' in import_line:
        print("✓ check_tiff_dtype already in import")
        
        # Verify it's correctly formatted
        if import_line.count('check_tiff_dtype') == 1:
            print("✓ Import looks correct")
            return True
        else:
            print("⚠ Multiple occurrences found, may need manual fix")
    
    # Fix the import line
    print("Fixing import line...")
    
    # Backup
    backup_path = filter_path + '.backup'
    shutil.copy(filter_path, backup_path)
    print(f"Backup saved to {backup_path}")
    
    # Ensure check_tiff_dtype is in the import
    if 'check_tiff_dtype' not in import_line:
        # Add check_tiff_dtype before return_to_original_dtype
        if 'return_to_original_dtype' in import_line:
            import_line = import_line.replace(
                'return_to_original_dtype',
                'check_tiff_dtype, return_to_original_dtype'
            )
        else:
            # Add at the end
            import_line = import_line.rstrip() + ', check_tiff_dtype\n'
        
        lines[import_line_idx] = import_line
        print(f"New import line: {import_line.strip()}")
        
        # Write back
        with open(filter_path, 'w') as f:
            f.writelines(lines)
        
        print("✓ Fixed!")
        return True
    
    return False

def clear_python_cache():
    """Clear Python cache files"""
    print("\nClearing Python cache files...")
    cache_dirs = []
    
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            cache_dirs.append(cache_path)
            try:
                shutil.rmtree(cache_path)
                print(f"  Removed: {cache_path}")
            except Exception as e:
                print(f"  Warning: Could not remove {cache_path}: {e}")
    
    # Also remove .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                try:
                    os.remove(pyc_path)
                    print(f"  Removed: {pyc_path}")
                except Exception as e:
                    print(f"  Warning: Could not remove {pyc_path}: {e}")
    
    if cache_dirs:
        print(f"✓ Cleared {len(cache_dirs)} cache directories")
    else:
        print("✓ No cache files found")

def verify_fix():
    """Verify the fix works"""
    print("\nVerifying fix...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        from gpufish.functions.core.filter import log_filter
        print("✓ filter module imported successfully")
        
        # Check if check_tiff_dtype is available
        from gpufish.functions.core import filter as filter_module
        if hasattr(filter_module, 'check_tiff_dtype'):
            print("✓ check_tiff_dtype available in filter module")
            return True
        else:
            print("✗ check_tiff_dtype still not available")
            return False
            
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("FIXING check_tiff_dtype IMPORT ISSUE")
    print("=" * 60)
    
    # Step 1: Fix imports
    if not fix_filter_imports():
        print("\n⚠ Could not fix imports automatically")
        print("Please check the import line manually")
    
    # Step 2: Clear cache
    clear_python_cache()
    
    # Step 3: Verify
    if verify_fix():
        print("\n" + "=" * 60)
        print("✓ FIX VERIFIED - Issue should be resolved!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠ Fix applied but verification failed")
        print("Please run your code again to test")
        print("=" * 60)
