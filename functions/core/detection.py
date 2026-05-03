from gpufish.functions.core.filter import log_filter
from tqdm import tqdm
from cucim.skimage.measure import label, regionprops
import numpy as np
from sklearn.cluster import DBSCAN
from gpufish.functions.core.parameters_calculation import compute_dbscan_eps_pixels
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

    log_image = log_filter(image, log_kernel_size)

    cc = label(log_image)
    regions = regionprops(cc, intensity_image=log_image)

    # -----------------------------
    # STEP 1: threshold filtering
    # -----------------------------
    filtered_spots = []
    intensities = []

    for r in tqdm(regions, desc="Filtering spots"):
        center = tuple(np.round(r.centroid).astype(int))
        intensity = float(image[center])

        if threshold is None or intensity > threshold:
            filtered_spots.append(center)
            intensities.append(intensity)

    print(f"{len(filtered_spots)} spots passed threshold")

    if len(filtered_spots) == 0:
        return []

    coords = np.array(filtered_spots)
    intensities = np.array(intensities)

    # -----------------------------
    # STEP 2: DBSCAN MERGING (small spots)
    # -----------------------------
    if dbscan:

        eps = compute_dbscan_eps_pixels(voxel_size, spot_radius, mode="mean")

        db = DBSCAN(eps=eps, min_samples=1)
        labels = db.fit_predict(coords)

        # Optional recovery (keep if useful)
        labels = recover_large_clusters(
            coords=coords,
            intensities=intensities,
            labels=labels,
            voxel_size=voxel_size,
            eps_scale=3.0,
            large_cluster_min_points=30
        )

        merged_spots = []
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            cluster_coords = coords[idx]
            centroid = np.mean(cluster_coords, axis=0)
            merged_spots.append(tuple(np.round(centroid).astype(int)))

    else:
        merged_spots = filtered_spots

    print(f"{len(merged_spots)} spots after DBSCAN merging")
    # -----------------------------
    # STEP 3: LARGE REGION COLLAPSE (NEW SIMPLE METHOD)
    # -----------------------------
    
    large_spots = collapse_large_regions(
        image,
        percentile=95,
        min_size=800   # <-- MAIN PARAMETER TO TUNE
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

# -----------------------------
# LARGE CLUSTER RECOVERY (FIXED)
# -----------------------------
def recover_large_clusters(
        coords,
        intensities,
        labels,
        voxel_size,
        eps_scale=3.0,
        large_cluster_min_points=10
    ):

    coords = np.asarray(coords, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    voxel_size = np.asarray(voxel_size, dtype=float)
    labels = np.asarray(labels)

    # --- FIX 1: ensure correct shape ---
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    new_labels = labels.copy()
    next_label = int(labels.max()) + 1 if len(labels) > 0 else 0

    # --- FIX 2: safe broadcasting ---
    coords_phys = coords * voxel_size.reshape(1, -1)

    unique_labels = np.unique(labels)

    for lab in unique_labels:

        if lab == -1:
            continue

        idx = np.where(labels == lab)[0]

        # --- FIX 3: skip empty clusters ---
        if len(idx) == 0:
            continue

        cluster_size = len(idx)
        cluster_intensity = np.mean(intensities[idx])

        if (cluster_size >= large_cluster_min_points or
            cluster_intensity > np.percentile(intensities, 90)):

            sub_coords = coords_phys[idx]

            # --- FIX 4: ensure 2D ---
            if sub_coords.ndim == 1:
                sub_coords = sub_coords.reshape(1, -1)

            eps = float(np.mean(voxel_size) * eps_scale)

            sub_db = DBSCAN(eps=eps, min_samples=1)
            sub_labels = sub_db.fit_predict(sub_coords)

            for sub in np.unique(sub_labels):
                sub_mask = (sub_labels == sub)

                # --- FIX 5: safe indexing ---
                if not np.any(sub_mask):
                    continue

                sub_idx = idx[sub_mask]

                new_labels[sub_idx] = next_label
                next_label += 1

    return new_labels
def collapse_large_regions(image, percentile=99, min_size=1000):
    """
    Find large bright connected regions and collapse each into one spot.
    """

    mask = image > np.percentile(image, percentile)

    cc = label(mask)
    regions = regionprops(cc, intensity_image=image)

    large_spots = []

    for r in regions:
        # region size in pixels
        if r.area >= min_size:

            # collapse to ONE representative point
            center = tuple(np.round(r.centroid).astype(int))
            large_spots.append(center)

    return large_spots