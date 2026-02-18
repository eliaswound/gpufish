from .filter import log_filter
from .filter import local_maximum_filter
from .conventional_threshold import _get_candidate_thresholds
from .conventional_threshold import _get_spot_threshold
from .conventional_threshold import spots_thresholding
def detect_spots(
        image,
        threshold=None,
        return_threshold=False,
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
    pixel_values = []
    masks = []
    image_filtered = []
    print("detecting spots in image")
    for item in image:
        # filter image using log filter
        image_filtered = stack.log_filter(item, log_kernel_size)
        image_filtered.append(image_filtered)
        # get pixels value
        pixel_values += list(image_filtered.ravel())
        # find local maximum using local maximum filter
        mask_local_max = local_maximum_filter(image_filtered, minimum_distance)
        masks.append(mask_local_max)
    if threshold is None:
        print("no threshold provided, finding threshold now")
         thresholds = _get_candidate_thresholds(pixel_values)
         threshold = _get_spot_threshold(thresholds)
         print("threshold: ", threshold)
    else: 
        print(" using provided threshold: ", threshold)
    all_spots = []
    for i in range(n):
        # get image and mask
        image_filtered = images_filtered[i]
        mask_local_max = masks[i]
 # detect spots using conventional thresholding
        spots, _ = spots_thresholding(
            image_filtered, mask_local_max, threshold)
        all_spots.append(spots)

    # return threshold or not
    if return_threshold:
        return all_spots, threshold
    else:
        return all_spots