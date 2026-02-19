from ..checks.check_image import check_cupy_array
from ..checks.check_image import check_tiff_dtype
from tqdm import tqdm
from cucim.skimage.measure import label, regionprops
import cupy as cp
import numpy as np
import warnings

def _get_candidate_thresholds(pixel_values):
    """Choose the candidate thresholds to test for the spot detection.

    Parameters
    ----------
    pixel_values : cp.ndarray
        Pixel intensity values of the image.

    Returns
    -------
    thresholds :np.ndarray, np.float64
        Candidate threshold values.

    """
    # choose appropriate thresholds candidate
    start_range = 0
    end_range = int(np.percentile(pixel_values, 99.9999))
    if end_range < 100:
        thresholds = np.linspace(start_range, end_range, num=100)
    else:
        thresholds = [i for i in range(start_range, end_range + 1)]
    thresholds = np.array(thresholds)

    return thresholds

def _get_spot_threshold(thresholds):
       
    all_value_spots = []
    minimum_threshold = float(thresholds[0])
    for i in range(n):
        image_filtered = images_filtered[i]
        mask_local_max = masks[i]
        spots, mask_spots = spots_thresholding(
            image_filtered, mask_local_max,
            threshold=minimum_threshold,
            remove_duplicate=False)
        value_spots = image_filtered[mask_spots]
        all_value_spots.append(value_spots)
    all_value_spots = np.concatenate(all_value_spots)
    thresholds, count_spots = _get_spot_counts(thresholds, all_value_spots)

    # select threshold where the kink of the distribution is located
    if count_spots.size > 0:
        threshold, _, _ = get_breaking_point(thresholds, count_spots)
    return threshold



def get_breaking_point(x, y):
    """Select the x-axis value where a L-curve has a kink.

    Assuming a L-curve from A to B, the 'breaking_point' is the more distant
    point to the segment [A, B].

    Parameters
    ----------
    x : np.array
        X-axis values.
    y : np.array
        Y-axis values.

    Returns
    -------
    breaking_point : float
        X-axis value at the kink location.
    x : np.array
        X-axis values.
    y : np.array
        Y-axis values.

    """
    # check parameters
    stack.check_array(
        x,
        ndim=1,
        dtype=[np.float32, np.float64, np.int32, np.int64])
    stack.check_array(
        y,
        ndim=1,
        dtype=[np.float32, np.float64, np.int32, np.int64])

    # select threshold where curve break
    slope = (y[-1] - y[0]) / len(y)
    y_grad = np.gradient(y)
    m = list(y_grad >= slope)
    j = m.index(False)
    m = m[j:]
    x = x[j:]
    y = y[j:]
    if True in m:
        i = m.index(True)
    else:
        i = -1
    breaking_point = float(x[i])

    return breaking_point, x, y

def spots_thresholding(
        image,
        mask_local_max,
        threshold):

    check_cupy_array(image)
    check_tiff_dtype(image)

    # remove peaks with a low intensity
    mask = (mask_local_max & (image > threshold))

    if mask.sum() == 0:
        spots = cp.array([], dtype=cp.int64).reshape((0, image.ndim))
        return spots, mask

    # when several pixels are assigned to the same spot, keep the centroid
    cc, _ = label(mask)

    local_max_regions = regionprops(cc)

    spots = []
    for local_max_region in tqdm(local_max_regions,
                                 desc="checking for duplicated spots"):
        spot = cp.array(local_max_region.centroid)
        spots.append(spot)

    spots = cp.stack(spots).astype(cp.int64)

    # build mask again
    mask = cp.zeros_like(mask)
    mask[tuple(spots.T)] = True

    if spots.size == 0:
        warnings.warn(
            "No spots were detected (threshold is {0}).".format(threshold),
            UserWarning
        )

    return spots, mask

