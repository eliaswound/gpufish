import numpy as np
import cupy as cp
from tqdm import tqdm
from cucim.skimage.measure import label, regionprops
from gpufish.functions.core.filter import log_filter, local_maximum_filter
from scipy import stats
import warnings
from skimage.measure import label as sk_label, regionprops as sk_regionprops
from gpufish.functions.core.parameters_calculation import compute_dbscan_eps_pixels, compute_radial_sym, fit_gaussian, compute_contrast, compute_zscore, roundness_from_3d_region
from gpufish.functions.core.detection import collapse_large_regions, recover_large_clusters 
from sklearn.cluster import DBSCAN

def regionprop_test_for_thresholds(
        image,
        regionprop_names=["mean_intensity","radial_symmetry","gaussian_fit","spot_count"],
        num_bins=None,
        thresholds=None,
        log_kernel_size=None,
        voxel_size=None,
        spot_radius=None,
        min_volume_thresh=0.6,
        threshold_range=None,
        dbscan=True
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
    cc = label(log_image)

    warnings.filterwarnings("ignore")
    regions = regionprops(cc, intensity_image=log_image)
    regular_regions = regionprops(cc, intensity_image=image)
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

    # -----------------------------
    # Optional pre-merge with DBSCAN
    # -----------------------------
    if dbscan:
        regions = list(regions)
        regular_regions = list(regular_regions)

        candidate_idx = []
        candidate_coords = []
        candidate_intensities = []

        for i, (r, rr) in enumerate(zip(regions, regular_regions)):
            if r.area < min_volume_pixels:
                continue

            c = np.asarray(r.centroid, dtype=float)
            if not np.all(np.isfinite(c)):
                continue

            coord = tuple(np.round(c).astype(int))
            if any(idx < 0 or idx >= image.shape[d] for d, idx in enumerate(coord)):
                continue

            if xp is cp:
                center_intensity = float(cp.asnumpy(image[coord]))
            else:
                center_intensity = float(image[coord])

            candidate_idx.append(i)
            candidate_coords.append(coord)
            candidate_intensities.append(center_intensity)

        if len(candidate_coords) > 0:
            coords_np = np.asarray(candidate_coords, dtype=float)
            intensities_np = np.asarray(candidate_intensities, dtype=float)

            eps = compute_dbscan_eps_pixels(voxel_size, spot_radius, mode="mean")
            labels = DBSCAN(eps=eps, min_samples=1).fit_predict(coords_np)

            kept_global_idx = []
            for lab in np.unique(labels):
                local_ids = np.where(labels == lab)[0]
                best_local = local_ids[np.argmax(intensities_np[local_ids])]
                kept_global_idx.append(candidate_idx[best_local])

            regions = [regions[i] for i in kept_global_idx]
            regular_regions = [regular_regions[i] for i in kept_global_idx]
        else:
            regions = []
            regular_regions = []

        print(f"{len(regions)} regions after DBSCAN merge")
    
    all_bin_results = {}
    all_t_tests = {}

    for regionprop_name in regionprop_names:
        centers = []
        values = []

        rp = regionprop_name.lower()

        for r, rr in tqdm(
            zip(regions, regular_regions),
            total=len(regions),
            desc=f"Processing regions for '{regionprop_name}'"
        ):
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
                    value = center_intensity / float(rr.mean_intensity)

                elif rp in ["exceeding", "center-mean"]:
                    value = center_intensity - rr.mean_intensity
                

                elif rp == "weighted_centroid_distance":
                    if not hasattr(rr, "weighted_centroid"):
                        continue
                    wc = cp.asarray(rr.weighted_centroid)
                    c  = cp.asarray(rr.centroid)
                    if not (np.all(np.isfinite(wc)) and np.all(np.isfinite(c))):
                        continue
                    value = float(cp.linalg.norm(wc - c).get())

                elif rp in ["convex_area", "solidity"]:
                    if rr.area < 4:
                        continue
                    value = getattr(rr, rp)
                elif rp == "contrast":
                    value = compute_contrast(rr, image)
                elif rp == "zscore":
                    value = compute_zscore(rr, image)
                # --- New regionprops ---
                elif rp == "radial_symmetry":
                    value = compute_radial_sym(rr.intensity_image)

                elif rp == "gaussian_fit":
                    value = fit_gaussian(rr.intensity_image)["sigma_avg"]

                elif rp == "spot_count":
                    value = 1  # each region counts as one spot
                elif rp == "roundness":
                    if rr.perimeter > 0:
                       value = roundness_from_3d_region(rr)
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
        clean_centers = []
        clean_values = []

        for c, v in zip(centers, values):
            # CuPy or NumPy scalars or Python floats/ints
            if isinstance(v, (int, float, np.floating, np.integer, cp.floating, cp.integer)):
                clean_centers.append(c)
                clean_values.append(float(v))
            # Optionally handle 0-dim CuPy / NumPy arrays
            elif hasattr(v, 'shape') and v.shape == ():
                clean_centers.append(c)
                clean_values.append(float(v.item()))
            else:
                # skip this region entirely if value is non-scalar
                continue

        centers = xp.asarray(clean_centers)
        values  = xp.asarray(clean_values)
        bin_results = {}
        for t in tqdm(thresholds, desc=f"Binning {regionprop_name}"):
            key = f"{int(float(t))}"
            mask = centers >= t
            if rp == "spot_count":
                bin_results[key] = int(np.sum(mask))
            else:
                bin_results[key] = values[mask] if np.any(mask) else xp.array([])

        # --- Welch t-tests ---
        t_tests = {}
        keys = list(bin_results.keys())
        for i in tqdm(range(len(keys) - 1), desc=f"T-tests {regionprop_name}"):
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

