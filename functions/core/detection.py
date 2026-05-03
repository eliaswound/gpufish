from gpufish.functions.core.filter import log_filter
from gpufish.functions.core.filter import local_maximum_filter  
from cucim.skimage.measure import label, regionprops
import numpy as np
from tqdm import tqdm

# -----------------------------
# EPS COMPUTATION (FIXED)
# -----------------------------
def compute_merge_radius_pixels(voxel_size, spot_radius, mode="mean"):
    """
    Convert nm spot radius -> merge radius in PIXEL space.
    """

    voxel_size = np.array(voxel_size)
    spot_radius = np.array(spot_radius)

    if mode == "vector":
        r_nm = spot_radius
    elif mode == "mean":
        r_nm = np.mean(spot_radius)
    elif mode == "max":
        r_nm = np.max(spot_radius)
    else:
        raise ValueError("mode must be 'vector', 'mean' or 'max'")

    # nm → voxel units per axis
    r_vox = r_nm / voxel_size

    if mode == "vector":
        return np.asarray(r_vox, dtype=float)

    # isotropic collapse for scalar radius
    return float(np.linalg.norm(r_vox))


def _to_numpy(array_like):
    """Convert cupy/numpy arrays to a numpy ndarray."""
    if hasattr(array_like, "get"):
        return array_like.get()
    return np.asarray(array_like)


def _ensure_min_distance_tuple(min_distance, ndim):
    """
    Convert min_distance to tuple format accepted by check_min_distance.
    """
    arr = np.asarray(min_distance).astype(float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, ndim)
    if arr.size != ndim:
        raise ValueError(
            f"min_distance has {arr.size} values but image has {ndim} dimensions."
        )
    arr = np.maximum(np.ceil(arr).astype(int), 1)
    return tuple(int(v) for v in arr)


def _squared_anisotropic_distance(points, center, radii_pixels):
    """
    Compute squared anisotropic distance in normalized pixel space.
    Distance <= 1 means the point is inside the merge ellipsoid.
    """
    safe_radii = np.maximum(np.asarray(radii_pixels, dtype=float), 1e-6)
    diff = (points - center) / safe_radii.reshape(1, -1)
    return np.sum(diff * diff, axis=1)


def merge_spots_without_dbscan(coords, intensities, voxel_size, spot_radius):
    """
    Greedy non-maximum suppression in anisotropic metric space.
    Keeps brightest point and removes neighbors within spot radius.
    """
    coords = np.asarray(coords, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if len(coords) == 0:
        return []

    radii_pixels = compute_merge_radius_pixels(voxel_size, spot_radius, mode="vector")
    order = np.argsort(intensities)[::-1]
    suppressed = np.zeros(len(coords), dtype=bool)
    kept_centers = []

    for idx in order:
        if suppressed[idx]:
            continue

        center = coords[idx]
        kept_centers.append(tuple(np.round(center).astype(int)))

        d2 = _squared_anisotropic_distance(coords, center, radii_pixels)
        suppressed[d2 <= 1.0] = True

    return kept_centers


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def detect_spots_threshold(
        image,
        threshold=None,
        voxel_size=None,
        spot_radius=None,
        log_kernel_size=None,
        dbscan=True
        ):

    print("Detecting spots in image")

    if voxel_size is None or spot_radius is None:
        raise ValueError("voxel_size and spot_radius are required for spot merging.")

    log_image = log_filter(image, log_kernel_size)
    log_image_np = _to_numpy(log_image)
    image_np = _to_numpy(image)

    min_distance = np.ceil(
        compute_merge_radius_pixels(voxel_size, spot_radius, mode="vector")
    ).astype(int)
    min_distance = _ensure_min_distance_tuple(min_distance, image_np.ndim)

    peak_mask = local_maximum_filter(log_image, min_distance=min_distance)
    peak_mask_np = _to_numpy(peak_mask) & (log_image_np > 0)

    cc = label(peak_mask_np)
    regions = regionprops(cc, intensity_image=image_np)

    # -----------------------------
    # STEP 1: threshold filtering
    # -----------------------------
    filtered_spots = []
    intensities = []

    for r in tqdm(regions, desc="Filtering spots"):
        center = tuple(np.round(r.centroid).astype(int))
        intensity = float(image_np[center])

        if threshold is None or intensity > threshold:
            filtered_spots.append(center)
            intensities.append(intensity)

    print(f"{len(filtered_spots)} spots passed threshold")

    if len(filtered_spots) == 0:
        return []

    coords = np.array(filtered_spots)
    intensities = np.array(intensities)

    # -----------------------------
    # STEP 2: NON-DBSCAN MERGING (small spots)
    # -----------------------------
    if dbscan:
        # Backward compatibility: keep parameter name but disable DBSCAN usage.
        merged_spots = merge_spots_without_dbscan(
            coords=coords,
            intensities=intensities,
            voxel_size=voxel_size,
            spot_radius=spot_radius
        )
    else:
        merged_spots = filtered_spots

    print(f"{len(merged_spots)} spots after non-DBSCAN merging")
    # -----------------------------
    # STEP 3: LARGE REGION COLLAPSE (NEW SIMPLE METHOD)
    # -----------------------------
    
    large_spots = collapse_large_regions(
        image_np,
        percentile=95,
        min_size=800,
        min_peak_distance=min_distance
    )
    
    print(f"{len(large_spots)} large regions collapsed into spots")
    # -----------------------------
    # STEP 4: COMBINE (avoid duplicates)
    # -----------------------------
    final_spots = merged_spots.copy()

    if len(merged_spots) > 0 and len(large_spots) > 0:
        merged_arr = np.array(merged_spots)

        for ls in large_spots:
            dists = np.linalg.norm(merged_arr - np.array(ls), axis=1)

            # only add if not already close to an existing spot
            if np.min(dists) > 5:  # distance threshold in pixels (tune)
                final_spots.append(ls)
    else:
        final_spots += large_spots

    print(f"{len(final_spots)} total spots after adding large blobs")

    return final_spots

def collapse_large_regions(image, percentile=99, min_size=1000, min_peak_distance=None):
    """
    Find large bright connected regions and recover one or more peak spots.
    """

    mask = image > np.percentile(image, percentile)

    cc = label(mask)
    regions = regionprops(cc, intensity_image=image)

    large_spots = []

    for r in regions:
        if r.area >= min_size:
            region_mask = cc == r.label
            region_image = np.where(region_mask, image, 0)

            # Use local maxima to recover multiple peaks in dense blobs.
            if min_peak_distance is None:
                peak_distance = tuple([2] * image.ndim)
            else:
                peak_distance = _ensure_min_distance_tuple(min_peak_distance, image.ndim)

            local_peaks = local_maximum_filter(region_image, min_distance=peak_distance)
            local_peaks = _to_numpy(local_peaks) & region_mask & (region_image > 0)

            peak_cc = label(local_peaks)
            peak_regions = regionprops(peak_cc, intensity_image=region_image)

            if len(peak_regions) == 0:
                center = tuple(np.round(r.centroid).astype(int))
                large_spots.append(center)
            else:
                for peak_r in peak_regions:
                    center = tuple(np.round(peak_r.weighted_centroid).astype(int))
                    large_spots.append(center)

    return large_spots