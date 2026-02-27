import numpy as np
import cupy as cp
from tqdm import tqdm
from cucim.skimage.measure import label, regionprops
from gpufish.functions.core.filter import log_filter, local_maximum_filter
from scipy import stats
import warnings
from scipy.optimize import curve_fit

def regionprop_test_for_thresholds(
        image,
        regionprop_names=["mean_intensity","radial_symmetry","gaussian_fit","spot_count"],
        num_bins=None,
        thresholds=None,
        log_kernel_size=None,
        minimum_distance=None,
        voxel_size=None,
        spot_radius=None,
        min_volume_thresh=0.6,
        threshold_range=None
):
    """
    Threshold binning for multiple regionprops with optional t-tests.
    Supports: 'mean_intensity', 'radial_symmetry', 'gaussian_fit', 'spot_count',
    'weighted_centroid_distance', etc.
    """

    if isinstance(regionprop_names, str):
        regionprop_names = [regionprop_names]

    # Backend detection (numpy / cupy)
    try:
        xp = cp.get_array_module(image)
    except Exception:
        xp = np

    # Spot detection
    log_image = log_filter(image, log_kernel_size)
    mask_local_max = local_maximum_filter(log_image, minimum_distance)
    cc = label(mask_local_max)

    warnings.filterwarnings("ignore")
    regions = regionprops(cc, intensity_image=image)

    if voxel_size is None or spot_radius is None:
        raise ValueError("voxel_size and spot_radius must be provided")

    # Compute minimum spot volume in pixels
    spot_radius_pixel = xp.array(spot_radius) / xp.array(voxel_size)
    theoretical_volume = 4/3 * np.pi * float(spot_radius_pixel[0]) * float(spot_radius_pixel[1]) * float(spot_radius_pixel[2])
    min_volume_pixels = min_volume_thresh * theoretical_volume

    print(f"Theoretical spot volume (pixels): {theoretical_volume:.2f}")
    print(f"Minimum accepted volume ({min_volume_thresh*100:.1f}%): {min_volume_pixels:.2f}")

    # Threshold generation
    if thresholds is None:
        if num_bins is None:
            raise ValueError("Provide num_bins or thresholds")
        min_val, max_val = threshold_range if threshold_range is not None else float(xp.min(image)), float(xp.max(image))
        thresholds = xp.linspace(min_val, max_val, num_bins, endpoint=False)
    thresholds = xp.asarray(thresholds)

    all_bin_results = {}
    all_t_tests = {}

    for regionprop_name in regionprop_names:
        centers = []
        values = []

        rp = regionprop_name.lower()

        for r in tqdm(regions, desc=f"Processing regions for '{regionprop_name}'"):
            # Volume filter
            if r.area < min_volume_pixels:
                continue

            # Safe centroid handling
            c = np.array(r.centroid)
            if not np.all(np.isfinite(c)):
                continue

            coords = tuple(np.round(c).astype(int))
            if any(idx < 0 or idx >= image.shape[i] for i, idx in enumerate(coords)):
                continue

            center_intensity = float(image[coords])

            try:
                # --- Existing regionprops ---
                if rp == "sbr":
                    if r.mean_intensity <= 0:
                        continue
                    value = center_intensity / float(r.mean_intensity)

                elif rp in ["exceeding", "center-mean"]:
                    value = center_intensity - r.mean_intensity
                

                elif rp == "weighted_centroid_distance":
                    if not hasattr(r, "weighted_centroid"):
                        continue
                    wc = cp.asarray(r.weighted_centroid)
                    c  = cp.asarray(r.centroid)
                    if not (np.all(np.isfinite(wc)) and np.all(np.isfinite(cu))):
                        continue
                    value = float(cp.linalg.norm(wc - c).get())

                elif rp in ["convex_area", "solidity"]:
                    if r.area < 4:
                        continue
                    value = getattr(r, regionprop_name)

                # --- New regionprops ---
                elif rp == "radial_symmetry":
                    value = compute_radial_sym(r.intensity_image)

                elif rp == "gaussian_fit":
                    value = fit_gaussian(r.intensity_image)["sigma_avg"]

                elif rp == "spot_count":
                    value = 1  # each region counts as one spot
                elif rp == "roundness":
                    if r.perimeter > 0:
                        value = 4 * np.pi * r.area / (r.perimeter ** 2)
                    else:
                        value = np.nan
                else:
                    if not hasattr(r, regionprop_name):
                        continue
                    value = getattr(r, regionprop_name)

            except Exception:
                continue

            if not np.isfinite(value):
                continue

            centers.append(center_intensity)
            values.append(value)

        if len(centers) == 0:
            print(f"Warning: No valid regions for '{regionprop_name}'. Skipping.")
            continue

        # Convert to arrays
        centers = xp.asarray(centers)
        clean_values = []
        for v in values:
            # CuPy or NumPy scalars or Python floats/ints
            if isinstance(v, (int, float, np.floating, np.integer, cp.floating, cp.integer)):
                clean_values.append(float(v))  # convert everything to plain float
            # Optionally handle 0-dim CuPy / NumPy arrays
            elif hasattr(v, 'shape') and v.shape == ():
                clean_values.append(float(v.item()))
            else:
                # skip anything else (non-scalar)
                continue

        # Now safe to convert to array
        values = xp.asarray(clean_values)

        # --- Cumulative binning ---
        bin_results = {}
        for t in thresholds:
            key = f"{int(float(t))}"

            if rp == "mean_intensity":
                mask = centers >= t
            else:
                mask = values >= t

            if rp == "spot_count":
                bin_results[key] = int(np.sum(mask))
            else:
                bin_results[key] = values[mask] if np.any(mask) else xp.array([])

        # --- Welch t-tests ---
        t_tests = {}
        keys = list(bin_results.keys())
        for i in range(len(keys) - 1):
            if rp == "spot_count":
                t_stat, p_value = 1.0, 1.0
            else:
                a = bin_results[keys[i]]
                b = bin_results[keys[i + 1]]

                if xp is not np:
                    if isinstance(a, cp.ndarray):
                        a = cp.asnumpy(a)
                    if isinstance(b, cp.ndarray):
                        b = cp.asnumpy(b)

                a = np.atleast_1d(a)
                b = np.atleast_1d(b)

                if len(a) > 1 and len(b) > 1:
                    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
                else:
                    t_stat, p_value = np.nan, np.nan

            t_tests[f"{keys[i]} vs {keys[i+1]}"] = (t_stat, p_value)

        # --- Save results ---
        all_bin_results[regionprop_name] = bin_results
        all_t_tests[regionprop_name] = t_tests

    return all_bin_results, all_t_tests

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