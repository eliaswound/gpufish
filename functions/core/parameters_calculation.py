def compute_radial_sym(intensity_image):
    """
    Compute a simple radial symmetry metric for a 3D spot.
    
    Parameters
    ----------
    intensity_image : ndarray
        3D array of spot intensities (from regionprops.intensity_image)
        
    Returns
    -------
    radial_sym : float
        Radial symmetry score in [0,1], 1 = perfectly symmetric
    """
    if intensity_image.size == 0:
        return np.nan

    # Get shape
    z_size, y_size, x_size = intensity_image.shape
    cz, cy, cx = (np.array(intensity_image.shape) - 1) / 2  # approximate centroid at center

    # Create coordinate grids
    zz, yy, xx = np.meshgrid(np.arange(z_size), np.arange(y_size), np.arange(x_size), indexing='ij')
    distances = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)

    # Flatten arrays
    distances = distances.flatten()
    intensities = intensity_image.flatten()

    if np.sum(intensities) == 0:
        return np.nan

    # Bin distances (radial bins)
    num_bins = max(3, int(np.ceil(np.max(distances))))
    radial_bins = np.linspace(0, np.max(distances), num_bins + 1)
    bin_indices = np.digitize(distances, radial_bins) - 1

    radial_mean = np.zeros(num_bins)
    radial_std = np.zeros(num_bins)

    for i in range(num_bins):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            radial_mean[i] = 0
            radial_std[i] = 0
        else:
            vals = intensities[mask]
            radial_mean[i] = np.mean(vals)
            radial_std[i] = np.std(vals)

    # Avoid divide by zero
    radial_mean[radial_mean == 0] = np.nan

    # Radial symmetry score: 1 - mean(CV) across radial bins
    radial_cv = radial_std / radial_mean
    radial_sym = 1 - np.nanmean(radial_cv)

    # Clamp to [0,1]
    radial_sym = max(0.0, min(1.0, radial_sym))

    return radial_sym


def fit_gaussian(intensity_image):
    import numpy as np
    from scipy.optimize import curve_fit

    # Move to CPU if needed
    intensity_image = intensity_image.get() if hasattr(intensity_image, "get") else intensity_image

    if intensity_image.size <= 1:
        return {'amplitude': np.nan, 'sigma_x': np.nan, 'sigma_y': np.nan,
                'background': np.nan, 'sigma_avg': np.nan}

    # Maximum-intensity z-slice
    z_sum = intensity_image.sum(axis=(1,2))
    z_idx = np.argmax(z_sum)
    slice2d = intensity_image[z_idx]

    if slice2d.max() <= 0:
        return {'amplitude': np.nan, 'sigma_x': np.nan, 'sigma_y': np.nan,
                'background': np.nan, 'sigma_avg': np.nan}

    # Coordinates
    y = np.arange(slice2d.shape[0])
    x = np.arange(slice2d.shape[1])
    xx, yy = np.meshgrid(x, y)

    xdata = np.vstack((xx.ravel(), yy.ravel()))
    ydata = slice2d.ravel()

    # 2D Gaussian
    def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, B):
        x, y = coords
        return A * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2))) + B

    # Initial guess
    A0 = slice2d.max() - slice2d.min()
    x0_0 = slice2d.shape[1] / 2
    y0_0 = slice2d.shape[0] / 2
    sigma_x0 = slice2d.shape[1] / 4
    sigma_y0 = slice2d.shape[0] / 4
    B0 = np.min(slice2d)
    p0 = (A0, x0_0, y0_0, sigma_x0, sigma_y0, B0)

    try:
        popt, _ = curve_fit(gaussian_2d, xdata, ydata, p0=p0, maxfev=5000)
        A, x0, y0, sigma_x, sigma_y, B = popt
        sigma_avg = (sigma_x + sigma_y) / 2
    except Exception:
        A = sigma_x = sigma_y = B = sigma_avg = np.nan

    return {'amplitude': A, 'sigma_x': sigma_x, 'sigma_y': sigma_y,
            'background': B, 'sigma_avg': sigma_avg}

 

def compute_contrast(region, image, ring_size=1):
    """
    Compute contrast = I_center - I_background_mean
    Works for single-pixel regions. CUDA-compatible.
    """
    # region mask on CPU
    region_mask = region.image
    if isinstance(region_mask, cp.ndarray):
        region_mask = cp.asnumpy(region_mask)
    else:
        region_mask = np.asarray(region_mask)

    # dilate to get background ring (CPU)
    if image.ndim == 2:
        dilated_mask = dilation(region_mask, square(2 * ring_size + 1))
    else:
        dilated_mask = dilation(region_mask, cube(2 * ring_size + 1))

    bg_mask = dilated_mask.astype(bool) & (~region_mask.astype(bool))

    # map to image coordinates
    if image.ndim == 2:
        minr, minc, maxr, maxc = region.bbox
        patch = image[minr:maxr, minc:maxc]
    else:
        minz, minr, minc, maxz, maxr, maxc = region.bbox
        patch = image[minz:maxz, minr:maxr, minc:maxc]

    if isinstance(patch, cp.ndarray):
        patch_cpu = cp.asnumpy(patch)
    else:
        patch_cpu = np.asarray(patch)

    bg_pixels_cpu = patch_cpu[bg_mask]
    if bg_pixels_cpu.size == 0:
        return cp.nan

    bg_pixels = cp.asarray(bg_pixels_cpu)

    I_center = float(region.max_intensity)
    I_bg_mean = float(cp.mean(bg_pixels))
    return float(I_center - I_bg_mean)


def compute_zscore(region, image, ring_size=1):
    """
    Compute z-score = (I_center - I_background_mean)/I_background_std
    CUDA-compatible: image can be NumPy or CuPy, stats done on GPU.
    """

    # --- Make region mask a NumPy array for skimage/scipy ---
    region_mask = region.image
    if isinstance(region_mask, cp.ndarray):
        region_mask = cp.asnumpy(region_mask)
    else:
        region_mask = np.asarray(region_mask)

    # Dilate to get background ring (CPU)
    if image.ndim == 2:
        dilated_mask = dilation(region_mask, square(2 * ring_size + 1))
    else:
        dilated_mask = dilation(region_mask, cube(2 * ring_size + 1))

    bg_mask = dilated_mask.astype(bool) & (~region_mask.astype(bool))

    # --- Map to image coordinates and extract background pixels on CPU ---
    if image.ndim == 2:
        minr, minc, maxr, maxc = region.bbox
        patch = image[minr:maxr, minc:maxc]
    else:
        minz, minr, minc, maxz, maxr, maxc = region.bbox
        patch = image[minz:maxz, minr:maxr, minc:maxc]

    # Ensure patch is NumPy before applying NumPy boolean mask
    if isinstance(patch, cp.ndarray):
        patch_cpu = cp.asnumpy(patch)
    else:
        patch_cpu = np.asarray(patch)

    bg_pixels_cpu = patch_cpu[bg_mask]
    if bg_pixels_cpu.size == 0:
        return cp.nan

    # Move background pixels to GPU for stats
    bg_pixels = cp.asarray(bg_pixels_cpu)

    if bg_pixels.size == 0 or cp.std(bg_pixels) == 0:
        return cp.nan

    I_center = float(region.max_intensity)
    return float((I_center - cp.mean(bg_pixels)) / cp.std(bg_pixels))



def roundness_from_3d_region(rr):
    """
    Compute 2D roundness for a 3D region by selecting the z-slice
    where the region has the largest area and computing
    4*pi*area / perimeter^2 on that slice.

    rr: cucim / skimage regionprops object for a 3D region
    """
    # rr.image is a 3D boolean array for the region (shape: z, y, x)
    mask3d = rr.image
    if mask3d.ndim != 3 or mask3d.size == 0:
        return np.nan

    # find the z-slice with the largest area
    slice_areas = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
    z_idx = int(np.argmax(slice_areas))
    mask2d = mask3d[z_idx]  # 2D binary mask (y, x)

    if mask2d.sum() == 0:
        return np.nan

    # label and compute regionprops on this 2D slice (CPU)
    labeled_2d = sk_label(mask2d.astype(bool))
    regions_2d = sk_regionprops(labeled_2d)

    if not regions_2d:
        return np.nan

    r2 = regions_2d[0]
    if r2.perimeter <= 0:
        return np.nan

    return 4.0 * np.pi * r2.area / (r2.perimeter ** 2)



# -----------------------------
# EPS COMPUTATION (FIXED)
# -----------------------------
def compute_dbscan_eps_pixels(voxel_size, spot_radius, mode="mean"):
    """
    Convert nm spot radius → DBSCAN eps in PIXEL space (anisotropy-aware).
    """

    voxel_size = np.array(voxel_size)
    spot_radius = np.array(spot_radius)

    if mode == "mean":
        r_nm = np.mean(spot_radius)
    elif mode == "max":
        r_nm = np.max(spot_radius)
    else:
        raise ValueError("mode must be 'mean' or 'max'")

    # nm → voxel units per axis
    r_vox = r_nm / voxel_size

    # isotropic collapse
    return np.linalg.norm(r_vox)

