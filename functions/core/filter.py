# Core functions for filtering images
# Log filter: Apply a Laplacian of Gaussian filter to a 2-d or 3-d image.
import numpy as np 
import cupy as cp 
from ..checks.check_image import check_cupy_array, fit_to_float, check_tiff_dtype, return_to_original_dtype
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