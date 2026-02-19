# Core functions for filtering images
# Log filter: Apply a Laplacian of Gaussian filter to a 2-d or 3-d image.
import numpy as np 
import cupy as cp 
# #region agent log
import json
import os
log_path = os.path.join(os.path.dirname(__file__), '..', '..', '.cursor', 'debug.log')
try:
    from ..checks.check_image import check_cupy_array, fit_to_float, check_tiff_dtype, return_to_original_dtype
    # Verify all imports succeeded
    imports_ok = all(name in globals() for name in ['check_cupy_array', 'fit_to_float', 'check_tiff_dtype', 'return_to_original_dtype'])
    with open(log_path, 'a') as f:
        f.write(json.dumps({"location":"filter.py:8","message":"Import check_image completed","data":{"all_imported":imports_ok,"check_tiff_dtype_in_globals":"check_tiff_dtype" in globals(),"check_tiff_dtype_callable":callable(check_tiff_dtype) if 'check_tiff_dtype' in globals() else False},"timestamp":int(__import__('time').time()*1000),"runId":"import_check","hypothesisId":"A"})+"\n")
    if not imports_ok:
        missing = [name for name in ['check_cupy_array', 'fit_to_float', 'check_tiff_dtype', 'return_to_original_dtype'] if name not in globals()]
        with open(log_path, 'a') as f:
            f.write(json.dumps({"location":"filter.py:8","message":"Some imports missing","data":{"missing":missing},"timestamp":int(__import__('time').time()*1000),"runId":"import_check","hypothesisId":"A"})+"\n")
except ImportError as e:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"location":"filter.py:8","message":"Import check_image failed","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(__import__('time').time()*1000),"runId":"import_check","hypothesisId":"A"})+"\n")
    raise
except Exception as e:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"location":"filter.py:8","message":"Unexpected error during import","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(__import__('time').time()*1000),"runId":"import_check","hypothesisId":"A"})+"\n")
    raise
# #endregion
from ..checks.check_inputs import check_sigma, check_min_distance
from cupyx.scipy.ndimage import gaussian_laplace, maximum_filter



def log_filter(image, sigma):
    """Apply a Laplacian of Gaussian filter to a 2-d or 3-d image.

    The function returns the inverse of the filtered image such that the pixels
    with the highest intensity from the original (smoothed) image have
    positive values. Those with a low intensity returning a negative value are
    clipped to zero.

    Parameters
    ----------
    image : cp.array
        Image with shape (z, y, x) or (y, x).
    sigma : int, float, Tuple(float, int) or List(float, int)
        Standard deviation used for the gaussian kernel (one for each
        dimension). If it's a scalar, the same standard deviation is applied
        to every dimensions.

    Returns
    -------
    image_filtered : cp.array 
        Filtered image.

    """
    # initialize and check inputs 
    # #region agent log
    import json
    import os
    log_path = os.path.join(os.path.dirname(__file__), '..', '..', '.cursor', 'debug.log')
    globals_dict = globals()
    locals_dict = locals()
    with open(log_path, 'a') as f:
        f.write(json.dumps({"location":"filter.py:50","message":"Before check_tiff_dtype call","data":{"check_tiff_dtype_in_locals":"check_tiff_dtype" in locals_dict,"check_tiff_dtype_in_globals":"check_tiff_dtype" in globals_dict,"available_globals":[k for k in globals_dict.keys() if 'check' in k.lower()]},"timestamp":int(__import__('time').time()*1000),"runId":"runtime_check","hypothesisId":"B"})+"\n")
    if 'check_tiff_dtype' not in globals():
        with open(log_path, 'a') as f:
            f.write(json.dumps({"location":"filter.py:50","message":"check_tiff_dtype NOT in globals - attempting reimport","data":{},"timestamp":int(__import__('time').time()*1000),"runId":"runtime_check","hypothesisId":"C"})+"\n")
        try:
            from ..checks.check_image import check_tiff_dtype
            globals()['check_tiff_dtype'] = check_tiff_dtype
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"filter.py:50","message":"Reimport successful","data":{},"timestamp":int(__import__('time').time()*1000),"runId":"runtime_check","hypothesisId":"C"})+"\n")
        except Exception as e:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"filter.py:50","message":"Reimport failed","data":{"error":str(e)},"timestamp":int(__import__('time').time()*1000),"runId":"runtime_check","hypothesisId":"C"})+"\n")
            raise NameError(f"check_tiff_dtype is not defined and reimport failed: {e}")
    # #endregion
    check_cupy_array(image)
    check_tiff_dtype(image)
    original_dtype, image_float = fit_to_float(image)
    check_sigma(sigma)

    image_filtered = gaussian_laplace(image_float, sigma=sigma)
    image_filtered = cp.clip(-image_filtered, a_min=0, a_max=None)
    return_to_original_dtype(image_filtered, original_dtype)
    return image_filtered

def local_maximum_filter(image, min_distance):
    """Compute a mask to keep only local maximum, in 2-d and 3-d.

    #. We apply a multidimensional maximum filter.
    #. A pixel which has the same value in the original and filtered images
       is a local maximum.

    Several connected pixels can have the same value. In such a case, the
    local maximum is not unique.

    In order to make the detection robust, it should be applied to a
    filtered image (using :func:`bigfish.stack.log_filter` for example).

    Parameters
    ----------
    image : np.ndarray
        Image to process with shape (z, y, x) or (y, x).
    min_distance : int, float, Tuple(int, float), List(int, float)
        Minimum distance (in pixels) between two spots we want to be able to
        detect separately. One value per spatial dimension (zyx or yx
        dimensions). If it's a scalar, the same distance is applied to every
        dimensions.

    Returns
    -------
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    """
    check_cupy_array(image)
    check_min_distance(min_distance)
    min_distance = np.ceil(min_distance).astype(image.dtype)
    kernel_size = 2 * min_distance + 1
    image_filtered = maximum_filter(image, size=kernel_size)
    mask = image == image_filtered
    
    return mask