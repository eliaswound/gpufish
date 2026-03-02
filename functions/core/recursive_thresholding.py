import numpy as np

def first_nonsignificant_bin(all_t_tests, rp_name, alpha=0.05):
    """
    Find the first bin pair (t_i vs t_{i+1}) that is NOT significant
    for a given regionprop and return the corresponding threshold interval
    (t_i, t_{i+1}).

    Parameters
    ----------
    all_t_tests : dict
        Output from regionprop_test_for_thresholds (second return value).
    rp_name : str
        Regionprop name, e.g. 'exceeding' or 'sbr'.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    interval : tuple[float, float] or None
        (t_i, t_{i+1}) for the first non-significant comparison,
        or None if all comparisons are significant or no tests exist.
    """
    rp_tests = all_t_tests.get(rp_name, {})
    if not rp_tests:
        return None

    parsed = []
    for comp_key, (t_stat, p_value) in rp_tests.items():
        try:
            left, right = comp_key.split(" vs ")
            # keys are e.g. "0", "3" or "0-3"; take the left number
            t0 = float(left.split("-")[0])
            t1 = float(right.split("-")[0])
        except Exception:
            continue
        parsed.append((t0, t1, comp_key, t_stat, p_value))

    if not parsed:
        return None

    # Sort by the lower threshold so "first" means lowest threshold region
    parsed.sort(key=lambda x: x[0])

    for _, (t0, t1, comp_key, t_stat, p_value) in enumerate(parsed):
        # non-significant: p >= alpha (or NaN p)
        if np.isnan(p_value) or p_value >= alpha:
            return (t0, t1)

    return None

def spot_count_slopes(all_bin_results):
    """
    Compute slopes of the spot_count curve between consecutive thresholds.

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

    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = dc / dt

    return thresholds, counts, slopes


def find_spot_count_plateau(all_bin_results, frac=1e-4):
    """
    Find the first threshold interval (t_i, t_{i+1}) where the absolute
    slope of spot_count falls below `frac` (default 0.01%) of the
    maximum absolute slope, and return (t_i, t_{i+1}).

    Parameters
    ----------
    all_bin_results : dict
        Output from regionprop_test_for_thresholds (first return value).
    frac : float
        Fraction of max |slope| to define the plateau (0.0001 = 0.01%).

    Returns
    -------
    interval : tuple[float, float] or None
        (t_i, t_{i+1}) where |slope| <= frac * max|slope|,
        or None if no such interval exists.
    """
    thresholds, counts, slopes = spot_count_slopes(all_bin_results)

    if slopes.size == 0 or np.all(np.isnan(slopes)):
        return None

    abs_slopes = np.abs(slopes)
    max_abs = np.nanmax(abs_slopes)
    if max_abs == 0 or np.isnan(max_abs):
        return None

    thr = max_abs * frac

    for i, s in enumerate(slopes):
        if np.isnan(s):
            continue
        if np.abs(s) <= thr:
            t0, t1 = thresholds[i], thresholds[i + 1]
            return (t0, t1)

    return None