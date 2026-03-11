import numpy as np


def summarize_spot_count_differences(all_bin_results):
    """
    Print spot counts per threshold and differences between consecutive bins.

    Expects that all_bin_results["spot_count"][key] is either:
    - an int
    - a NumPy scalar
    - a 0-D / length-1 array containing the count
    """
    if "spot_count" not in all_bin_results:
        print("No 'spot_count' entry in all_bin_results.")
        return

    bin_results = all_bin_results["spot_count"]

    # Sort keys numerically (keys are strings like "0", "3", "7", ...)
    keys = sorted(bin_results.keys(), key=lambda k: int(k.split("-")[0]))

    counts = []
    for k in keys:
        v = bin_results[k]
        # Handle scalar int, numpy scalar, or length-1 array
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) == 1:
            counts.append(int(v[0]))
        else:
            counts.append(int(v))

    counts = np.asarray(counts, dtype=int)

    print("Spot counts per threshold:")
    for k, c in zip(keys, counts):
        print(f"  threshold {k}: {c} spots")

    print("\nDifferences between consecutive thresholds (previous - current):")
    for prev_k, prev_c, curr_k, curr_c in zip(
        keys[:-1], counts[:-1], keys[1:], counts[1:]
    ):
        diff = prev_c - curr_c
        print(f"  {prev_k} -> {curr_k}: {prev_c} -> {curr_c}, diff = {diff}")


def spot_count_slopes(all_bin_results):
    """
    Compute slopes of the spot_count curve between consecutive thresholds.

    The slope between two consecutive thresholds t_i and t_{i+1} is:
        slope_i = (count_{i+1} - count_i) / (t_{i+1} - t_i)

    Returns
    -------
    thresholds : np.ndarray
        Sorted thresholds as floats.
    counts : np.ndarray
        Spot counts per threshold.
    slopes : np.ndarray
        Per-segment slopes between consecutive thresholds.
    """
    if "spot_count" not in all_bin_results:
        raise ValueError("No 'spot_count' entry in all_bin_results.")

    bin_results = all_bin_results["spot_count"]

    # Sort keys numerically (keys are strings like "0", "3", "7", ...)
    keys = sorted(bin_results.keys(), key=lambda k: int(k.split("-")[0]))

    thresholds = np.array([float(k.split("-")[0]) for k in keys], dtype=float)

    counts = []
    for k in keys:
        v = bin_results[k]
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) == 1:
            counts.append(int(v[0]))
        else:
            counts.append(int(v))

    counts = np.array(counts, dtype=float)

    if thresholds.size < 2:
        raise ValueError("Need at least two thresholds to compute slopes.")

    dt = np.diff(thresholds)
    dc = np.diff(counts)

    # Avoid division by zero if thresholds are duplicated
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = dc / dt

    # Optional: print a concise summary
    print("Threshold, Count:")
    for t, c in zip(thresholds, counts):
        print(f"  {t:.0f}: {int(c)}")

    print("\nSlopes between bins (Δcount / Δthreshold):")
    for t0, t1, s in zip(thresholds[:-1], thresholds[1:], slopes):
        print(f"  {t0:.0f} -> {t1:.0f}: slope = {s:.3f}")

    return thresholds, counts, slopes

