import numpy as np
import cupy as cp
from tqdm import tqdm
from cucim.skimage.measure import label, regionprops
from gpufish.functions.core.filter import log_filter, local_maximum_filter
from scipy import stats
import warnings

def regionprop_test_for_thresholds(
        image,
        regionprop_names=["mean_intensity"],
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
    Cumulative threshold binning with support for multiple regionprops.
    Supports:
        - mean_intensity
        - SBR
        - exceeding / center-mean
        - convex_area
        - solidity
        - weighted_centroid_distance
    """

    if isinstance(regionprop_names, str):
        regionprop_names = [regionprop_names]

    # Backend detection
    try:
        xp = cp.get_array_module(image)
    except Exception:
        xp = np

    # Spot detection
    log_image = log_filter(image, log_kernel_size)
    mask_local_max = local_maximum_filter(log_image, minimum_distance)
    cc = label(mask_local_max)

    warnings.filterwarnings("ignore")
    regions = regionprops(cc, intensity_image=log_image)

    # Volume filtering
    if voxel_size is None or spot_radius is None:
        raise ValueError("voxel_size and spot_radius must be provided")

    spot_radius_pixel = xp.array(spot_radius) / xp.array(voxel_size)
    theoretical_volume = 4/3 * np.pi * float(spot_radius_pixel[0]) * float(spot_radius_pixel[1]) * float(spot_radius_pixel[2])
    min_volume_pixels = min_volume_thresh * theoretical_volume

    print(f"Theoretical spot volume (pixels): {theoretical_volume:.2f}")
    print(f"Minimum accepted volume ({min_volume_thresh*100:.1f}%): {min_volume_pixels:.2f}")

    # Threshold generation
    if thresholds is None:
        if num_bins is None:
            raise ValueError("Provide num_bins or thresholds")
        if threshold_range is None:
            min_val, max_val = float(xp.min(image)), float(xp.max(image))
        else:
            min_val, max_val = threshold_range
        thresholds = xp.linspace(min_val, max_val, num_bins, endpoint=False)

    thresholds = xp.asarray(thresholds)

    all_bin_results = {}
    all_t_tests = {}

    for regionprop_name in regionprop_names:
        centers = []
        values = []

        for r in tqdm(regions, desc=f"Processing regions for '{regionprop_name}'"):

            # Area filter
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
            rp = regionprop_name.lower()

            try:
                if rp == "sbr":
                    if r.mean_intensity <= 0:
                        continue
                    value = center_intensity / float(r.mean_intensity)

                elif rp in ["exceeding", "center-mean"]:
                    # New formula: exclude center from the mean of rest of the region
                    if r.area <= 1:
                        continue
                    value = center_intensity - ((r.mean_intensity * r.area - center_intensity) / (r.area - 1))

                elif rp == "weighted_centroid_distance":
                    intensities = r.intensity_image[r.coords[:,0], r.coords[:,1]]
                    if np.sum(intensities) == 0:
                        continue

                    wc = np.average(r.coords, axis=0, weights=intensities)  # weighted centroid
                    value = np.linalg.norm(wc - c)
                    

                elif rp in ["convex_area", "solidity"]:
                    if r.area < 4:
                        continue
                    value = getattr(r, regionprop_name)

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

        centers = xp.asarray(centers)
        values = xp.asarray(values)

        # Cumulative binning
        bin_results = {}
        for t in thresholds:
            mask = centers >= t
            key = f"{int(float(t))}"
            bin_results[key] = values[mask]

        # Welch t-tests
        t_tests = {}
        keys = list(bin_results.keys())
        for i in range(len(keys) - 1):
            a = bin_results[keys[i]]
            b = bin_results[keys[i + 1]]
            if xp is not np:
                a = cp.asnumpy(a)
                b = cp.asnumpy(b)
            if len(a) > 1 and len(b) > 1:
                t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
            else:
                t_stat, p_value = np.nan, np.nan
            t_tests[f"{keys[i]} vs {keys[i+1]}"] = (t_stat, p_value)

        all_bin_results[regionprop_name] = bin_results
        all_t_tests[regionprop_name] = t_tests

    return all_bin_results, all_t_tests