# File IO for gpufish 
import tifffile as tiff
import cupy as cp
from ..checks.check_image import check_tiff_dtype
def read_tif(file_path):
    """
    Read a tif file and return a CuPy array.
    """
    array = tiff.imread(file_path)
    check_tiff_dtype(array)
    return cp.asarray(array) 


