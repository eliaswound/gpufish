# File IO for gpufish 
import tifffile as tiff
import cupy as cp
from gpufish.functions.checks.check_image import check_tiff_dtype, check_cupy_array, check_same_dtype
import numpy as np
import os
def read_tif(file_path):
    """
    Read a tif file and return a CuPy array.
    """
    array = tiff.imread(file_path)
    check_tiff_dtype(array)
    return cp.asarray(array) 

def overlay_spots(original_image, spots_image, alpha=0.5, save_image=True, save_path='./', save_name='overlay_semitransparent.tif'): 
    """
    Overlay the spots on the original image.
    If the inputs are CuPy arrays, they will be converted to NumPy arrays first.
    """
    check_cupy_array(original_image)
    original_image = cp.asnumpy(original_image)
    check_same_dtype(original_image, spots_image)
    # Clip and convert to uint8 for visualization
    check_tiff_dtype(original_image)
    check_tiff_dtype(spots_image)
    original_image = np.clip(original_image, 0, 255).astype(np.uint8)
    spots_image = np.clip(spots_image, 0, 255).astype(np.uint8)
    spots_scaled = (spots_image * alpha).astype(np.uint8)

    # Create RGB stack
    rgb_stack = np.zeros((original_image.shape[0], 3, original_image.shape[1], original_image.shape[2]), dtype=np.uint8)
    rgb_stack[:, 0, :, :] = original_image       # Red channel
    rgb_stack[:, 1, :, :] = np.maximum(rgb_stack[:, 1, :, :], spots_scaled)  # Green channel

    # Save if requested
    if save_image:
        tiff.imwrite(os.path.join(save_path, save_name), rgb_stack, photometric='rgb')

    return rgb_stack


