from gpufish.functions.core.filter import log_filter
from gpufish.functions.core.filter import local_maximum_filter
from tqdm import tqdm
from cucim.skimage.measure import label
from cucim.skimage.measure import regionprops
import numpy as np
def detect_spots(
        image,
        threshold=None,
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
    log_image = log_filter(image, log_kernel_size)
    mask_local_max = local_maximum_filter(log_image, minimum_distance)
    mask = (mask_local_max & (image > threshold))
    cc = label(mask)
    local_max_regions = regionprops(cc)
    spot = np.array([r.centroid for r in local_max_regions])
    return spot

<<<<<<< HEAD

=======
>>>>>>> 09667f58b6a81e7046ae11b0f6b23d73e95dbbe9
def detect_spots_exceeding(
        image,
        threshold=None,
        voxel_size=None,
        spot_radius=None,
        log_kernel_size=None,
        minimum_distance=None):
    """
    Detect spots in image and filter by exceeding value on LoG image.
    
    Parameters:
        image: np.array or torch.tensor, input image
        threshold: float, the exceeding threshold to filter spots
        voxel_size: optional, voxel size
        spot_radius: optional, spot radius
        log_kernel_size: optional, LoG kernel size
        minimum_distance: optional, minimum distance between spots
    Returns:
        filtered_spots: list of regionprops that passed the exceeding filter
    """
    print("Detecting spots in image")
    
    # 1. Apply LoG filter
    log_image = log_filter(image, log_kernel_size)
    
    # 2. Detect local maxima
    mask_local_max = local_maximum_filter(log_image, minimum_distance)
    mask = mask_local_max & (image > 0)
    
    # 3. Label connected components
    cc = label(mask)
<<<<<<< HEAD
    local_max_regions = regionprops(cc, intensity_image=log_image)
    
    # 4. Filter spots based on exceeding
    filtered_spots = []
    for r in tqdm(local_max_regions, desc = "Calculating exceedings"):
        center_coords = tuple(np.round(r.centroid).astype(int))
        center_intensity = float(image[center_coords])
=======
    local_max_regions = regionprops(cc, intensity_image=image)
    
    # 4. Filter spots based on exceeding
    filtered_spots = []
    for r in local_max_regions:
        center_coords = tuple(map(int, r.centroid))
        center_intensity = image[center_coords]
>>>>>>> 09667f58b6a81e7046ae11b0f6b23d73e95dbbe9
        mean_intensity = r.mean_intensity
        value = center_intensity - mean_intensity
        
        if threshold is None or value > threshold:
<<<<<<< HEAD
            filtered_spots.append(center_coords)
=======
            filtered_spots.append(r)
>>>>>>> 09667f58b6a81e7046ae11b0f6b23d73e95dbbe9
    
    print(f"{len(filtered_spots)} spots passed the exceeding threshold")
    return filtered_spots