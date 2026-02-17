# Core functions for filtering images
# Log filter: Apply a Laplacian of Gaussian filter to a 2-d or 3-d image.
import numpy as np 
import cupy as cp 
from ..checks.check_image import check_cupy_array, fit_to_float
from ..checks.check_inputs import check_sigma
from cupyx.scipy.ndimage import gaussian_laplace
from ..checks.check_image import return_to_original_dtype



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
    original_dtype, image_float = fit_to_float(image)
    check_sigma(sigma)

    image_filtered = gaussian_laplace(image_float, sigma=sigma)
    image_filtered = cp.clip(-image_filtered, a_min=0, a_max=None)
    return_to_original_dtype(image_filtered, original_dtype)
    return image_filtered
