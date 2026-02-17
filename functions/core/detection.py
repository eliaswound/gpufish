
def detect_spots(
        image,
        threshold=None,
        remove_duplicate=True,
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
    # Sanity check of image size 
    n = len(image)
    print("image size: ", image.shape)

    # apply LoG filter and find local maximum
    image_filtered = []
    pixel_values = []
    masks = []
    for image in images:
        # filter image
        image_filtered = stack.log_filter(image, log_kernel_size)
        images_filtered.append(image_filtered)
        # apply LoG filter and find local maximum
