import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math


def plot_all_threshod_test_results(
    all_bin_results,
    all_t_tests,
    alpha=0.05,
    log_threshold=100,
    save_image=False,
    save_path=None,
    max_bins=64
):
    """
    Fast plotting function.
    Uses precomputed t-tests.
    Pools bins only for visualization if bin count > max_bins.
    Disables significance stars when pooling occurs.
    """

    sns.set(style="whitegrid")

    for rp_name, bin_results in all_bin_results.items():

        # Convert CuPy → NumPy
        bin_results = {
            k: cp.asnumpy(v) if isinstance(v, cp.ndarray) else v
            for k, v in bin_results.items()
        }

        # Sort bins numerically
        bin_keys = sorted(bin_results.keys(), key=lambda x: int(x.split('-')[0]))
        data = [bin_results[k] for k in bin_keys]

        original_bin_count = len(data)
        pooled = False

        # -----------------------------
        # Pool bins if too many
        # -----------------------------
        if original_bin_count > max_bins:
            pooled = True
            pool_size = math.ceil(original_bin_count / max_bins)

            pooled_data = []
            pooled_keys = []

            for i in range(0, original_bin_count, pool_size):
                chunk = data[i:i + pool_size]

                # lightweight merge
                merged = np.concatenate(chunk) if len(chunk) > 0 else np.array([])
                pooled_data.append(merged)

                pooled_keys.append(bin_keys[i])

            data = pooled_data
            bin_keys = pooled_keys

        # -----------------------------
        # Remove negative values (clean log behavior)
        # -----------------------------
        data = [d[d > 0] if len(d) > 0 else np.array([1e-12]) for d in data]

        valid_arrays = [d for d in data if len(d) > 0]
        if len(valid_arrays) == 0:
            continue

        max_val = np.nanmax([np.nanmax(d) for d in valid_arrays])
        min_val = np.nanmin([np.nanmin(d) for d in valid_arrays])

        use_log = (max_val / (min_val + 1e-12)) > log_threshold

        # Fold tiny values upward
        epsilon = max_val * 1e-3
        data = [np.maximum(d, epsilon) for d in data]

        means = [np.nanmean(d) for d in data]

        # -----------------------------
        # Plot
        # -----------------------------
        # Dynamically scale figure width with number of bins so stars/labels spread out
        fig_width = min(40, max(12, len(bin_keys) * 0.25))
        plt.figure(figsize=(fig_width, 6), dpi=200)
        sns.boxplot(data=data)
        plt.plot(
            np.arange(len(means)),
            means,
            marker='o',
            color='red',
            linestyle='-',
            linewidth=2,
            label='Mean'
        )

        if use_log:
            plt.yscale('log')

        # -----------------------------
        # Add significance stars ONLY if not pooled
        # -----------------------------
        if not pooled:
            t_tests = all_t_tests.get(rp_name, {})

            y_max = np.nanmax([np.nanmax(d) for d in data])
            y_min = np.nanmin([np.nanmin(d) for d in data])
            step = (
                (np.log10(y_max + 1e-12) - np.log10(y_min + 1e-12)) * 0.05
                if use_log else (y_max - y_min) * 0.05
            )

            for i in range(len(bin_keys) - 1):

                key = f"{bin_keys[i]} vs {bin_keys[i+1]}"
                t_stat, p_value = t_tests.get(key, (np.nan, np.nan))

                if np.isnan(p_value):
                    continue

                if p_value < 0.001:
                    stars = '***'
                elif p_value < 0.01:
                    stars = '**'
                elif p_value < alpha:
                    stars = '*'
                else:
                    stars = None

                if stars:
                    x1, x2 = i, i + 1

                    # Push annotation further away from boxes for readability
                    y_pair_max = max(np.nanmax(data[i]), np.nanmax(data[i + 1]))
                    y = y_pair_max + 2 * step  # previously + step

                    plt.plot(
                        [x1, x1, x2, x2],
                        [y, y + step, y + step, y],
                        color='black'
                    )

                    plt.text(
                        (x1 + x2) / 2,
                        y + step,
                        stars,
                        ha='center',
                        va='bottom',
                        fontsize=8,  # smaller label for less clutter
                    )

            # Expand y-limits so stars are not stuck at top border
            ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            if use_log:
                ax.set_ylim(ymin, ymax * 1.3)
            else:
                ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.3)

        # -----------------------------
        # Final formatting
        # -----------------------------
        # X-axis: keep original bin labels, but show every other label to reduce crowding
        label_step = 2
        tick_positions = np.arange(0, len(bin_keys), label_step)
        tick_labels = [bin_keys[i] for i in tick_positions]

        plt.xticks(
            ticks=tick_positions,
            labels=tick_labels,
            rotation=0,
            ha='right',
            fontsize=8,
        )

        plt.xlabel("Intensity Bins")
        plt.ylabel(rp_name)
        title = f"Binned Analysis: {rp_name}"
        if pooled:
            title += f" (pooled to ≤{max_bins} bins)"
        plt.title(title)

        plt.legend()
        plt.tight_layout()

        # -----------------------------
        # Save if requested
        # -----------------------------
        if save_image and save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"{rp_name}_plot_with_different_threshold.jpg"
            plt.savefig(os.path.join(save_path, filename), dpi=300)

        plt.show()