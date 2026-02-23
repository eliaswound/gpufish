from gpufish.functions.core.filter import log_filter
from gpufish.functions.core.filter import local_maximum_filter
from .conventional_threshold import _get_candidate_thresholds
from .conventional_threshold import _get_spot_threshold
from .conventional_threshold import spots_thresholding
from tqdm import tqdm
from cucim.skimage.measure import label
from cucim.skimage.measure import regionprops
def regionprop_test(
        image,
        threshold_lower = 1,
        threshold_upper = 10,
        voxel_size=None,
        spot_radius=None,
        log_kernel_size=None,
        minimum_distance=None):
    """ detect spots in image using gpu if available, otherwise use cpu
    Parameters:
    image:single instance of image to detect spots in, input should be np.arrary or torch.tensor
    threshold: threshold for detecting spots, if not provided, a heuristic threshold will be used
    remove_duplicate: remove duplicate spots, default is True
    return_threshold: return the threshold used for detecting spots, default is False
    voxel_size: voxel size for detecting spots, default is None
    spot_radius: radius of spots, default is None
    log_kernel_size: size of the log kernel, default is None
    minimum_distance: minimum distance between spots, default is None
    """
    print("detecting spots in image")
    pixel_values = []
    log_image = log_filter(image, log_kernel_size)
    pixel_values += list(log_image.ravel())
    mask_local_max = local_maximum_filter(log_image, minimum_distance)
    lower_mask = (mask_local_max & (image > threshold_lower))
    upper_mask = (mask_local_max & (image > threshold_upper))
    lower_cc = label(lower_mask)
    upper_cc = label(upper_mask)
    lower_local_max_regions = regionprops(lower_cc, intensity_image=image)
    upper_local_max_regions = regionprops(upper_cc, intensity_image=image)
    lower_only_mask = lower_mask & (~upper_mask)

    # return threshold or not
    if return_threshold:
        return spot, threshold
    else:
        return spot