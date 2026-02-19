#!/bin/bash
# Fix script for Linux machine
# Run this on Linux: bash fix_imports_linux.sh

echo "Fixing check_tiff_dtype import issue..."

FILTER_FILE="functions/core/filter.py"

if [ ! -f "$FILTER_FILE" ]; then
    echo "Error: $FILTER_FILE not found!"
    exit 1
fi

# Check if check_tiff_dtype is in the import line
if grep -q "check_tiff_dtype" "$FILTER_FILE"; then
    echo "✓ check_tiff_dtype found in imports"
    
    # Check if it's actually in an import statement
    if grep -q "from ..checks.check_image import.*check_tiff_dtype" "$FILTER_FILE"; then
        echo "✓ Import statement looks correct"
    else
        echo "✗ check_tiff_dtype found but not in import statement!"
        echo "  Current import line:"
        grep "from ..checks.check_image import" "$FILTER_FILE" | head -1
        echo ""
        echo "  Fixing..."
        # This is a backup - manual fix needed
        echo "  Please manually verify the import line includes check_tiff_dtype"
    fi
else
    echo "✗ check_tiff_dtype NOT found in $FILTER_FILE"
    echo "  Adding to import..."
    
    # Find the import line and add check_tiff_dtype
    sed -i.bak 's/from ..checks.check_image import check_cupy_array, fit_to_float\(, return_to_original_dtype\)\?/from ..checks.check_image import check_cupy_array, fit_to_float, check_tiff_dtype, return_to_original_dtype/' "$FILTER_FILE"
    
    echo "  ✓ Fixed! Backup saved as $FILTER_FILE.bak"
fi

echo ""
echo "Verification:"
grep -n "from ..checks.check_image import" "$FILTER_FILE"

echo ""
echo "Done!"
