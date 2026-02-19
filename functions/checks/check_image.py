 # Check image and change image settings if needed
import numpy as np
import cupy as cp
def check_cupy_array(image):
    """
    Stop execution unless `image` is a CuPy ndarray.
    Parameters:
    image: any
        Input image (array like object) to check.
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise ValueError("can not import cupy, please install cupy or use gpu device") from e

    if not isinstance(image, cp.ndarray):
        raise ValueError("please input image as cupy array")


def check_tiff_dtype(image):
    """
    Check if the image array has a valid TIFF dtype.
    
    Valid dtypes: uint8, uint16, uint32, float8, float16, float32
    
    Parameters
    ----------
    image : array_like
        Input image array to check.
    
    Raises
    ------
    ValueError
        If image dtype is not one of the valid TIFF dtypes.
    """
    valid_dtypes = [
        np.uint8, np.uint16, np.uint32,
        np.float16, np.float32, np.float64
    ]
    
    # Handle both NumPy and CuPy arrays
    dtype = image.dtype
    
    if dtype not in valid_dtypes:
        raise ValueError("please input a correct tiff array, valid dtypes are uint8, uint16, uint32, float16, float32,float64")

def fit_to_float(image):
    """
    Convert image to appropriate float dtype based on input dtype.
    
    Parameters
    ----------
    image : cupy.ndarray
        Input CuPy array image.
    
    Returns
    -------
    tuple : (cupy.ndarray, dtype)
        Converted float array and original dtype.
    """
    # ensure input is a CuPy array
    check_cupy_array(image)
    
    # Store original dtype
    original_dtype = image.dtype
    
    # Conversion rules
    if original_dtype in [cp.uint8, cp.uint16, cp.float16]:
        converted_image = image.astype(cp.float32)
        return converted_image, original_dtype
    
    elif original_dtype == cp.float32:
        return image, original_dtype
    
    elif original_dtype == cp.uint32:
        converted_image = image.astype(cp.float64)
        return converted_image, original_dtype
    
    elif original_dtype == cp.float64:
        return image, original_dtype
    
    else:
        raise ValueError(f"Unsupported dtype: {original_dtype}")


def return_to_original_dtype(image, dtype):
    """
    Cast `image` back to `dtype` if needed.

    Parameters
    ----------
    image : array
        NumPy or CuPy array.
    dtype : numpy.dtype / cupy.dtype / str
        Target dtype (typically the original dtype you saved earlier).

    Returns
    -------
    array
        `image` unchanged if already in `dtype`, otherwise a casted copy.
    """
    if image.dtype == dtype:
        return image
    return image.astype(dtype)

def check_same_dtype(image1, image2):
    """
    Check if two images have the same data type.
    
    Parameters
    ----------
    image1 : array_like
        First image array to check.
    image2 : array_like
        Second image array to check.
    
    Raises
    ------
    ValueError
        If images do not have the same dtype.
    """
    dtype1 = image1.dtype
    dtype2 = image2.dtype
    
    if dtype1 != dtype2:
        print("image are not the same data type please check inputs")
        raise ValueError(f"Images have different dtypes: {dtype1} vs {dtype2}")