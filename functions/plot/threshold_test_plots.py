import seaborn as sns
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import os

def plot_all_threshod_test_results(
        all_bin_results,
        all_t_tests,
        alpha=0.05,
        log_threshold=100,
        save_image=False,
        save_path=None
):
    import os
    sns.set(style="whitegrid")

    for rp_name, bin_results in all_bin_results.items():

        # Convert CuPy arrays
        bin_results = {
            k: cp.asnumpy(v) if isinstance(v, cp.ndarray) else v
            for k, v in bin_results.items()
        }

        bin_keys = sorted(bin_results.keys(), key=lambda x: int(x.split('-')[0]))
        data = [bin_results[k] for k in bin_keys]

        # Compute raw min/max
        valid_arrays = [d for d in data if len(d) > 0]
        if len(valid_arrays) == 0:
            continue

        max_val = np.nanmax([np.nanmax(d) for d in valid_arrays])
        min_val = np.nanmin([np.nanmin(d) for d in valid_arrays])

        use_log = (max_val / (abs(min_val) + 1e-12)) > log_threshold

        # Clean data if log scale
        if use_log:
            data = [np.where(d > 1, d, 0) for d in data]

        means = [np.nanmean(d[d > 0]) if np.any(d > 0) else np.nan for d in data]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, palette="viridis")
        plt.plot(np.arange(len(means)), means,
                 marker='o', color='red',
                 linestyle='-', linewidth=2, label='Mean')

        if use_log:
            plt.yscale('log')

        # t-tests
        t_tests = all_t_tests.get(rp_name, {})

        valid_arrays = [d[d > 0] for d in data if np.any(d > 0)]
        if len(valid_arrays) > 0:
            y_max = np.nanmax([np.nanmax(d) for d in valid_arrays])
            y_min = np.nanmin([np.nanmin(d) for d in valid_arrays])
            step = (np.log10(y_max+1e-12) -
                    np.log10(y_min+1e-12))*0.05 if use_log else (y_max - y_min)*0.05
        else:
            step = 0

        for i in range(len(bin_keys) - 1):
            key = f"{bin_keys[i]} vs {bin_keys[i+1]}"
            t_stat, p_value = t_tests.get(key, (np.nan, np.nan))

            if np.isnan(p_value):
                continue

            if p_value < 0.001: stars = '***'
            elif p_value < 0.01: stars = '**'
            elif p_value < 0.05: stars = '*'
            else: stars = None

            if stars:
                x1, x2 = i, i+1
                a = data[i]
                b = data[i+1]

                a_pos = a[a > 0]
                b_pos = b[b > 0]

                if len(a_pos) == 0 or len(b_pos) == 0:
                    continue

                y = max(np.nanmax(a_pos),
                        np.nanmax(b_pos)) + step

                plt.plot([x1, x1, x2, x2],
                         [y, y+step, y+step, y],
                         color='black', linewidth=1)

                plt.text((x1+x2)/2, y+step,
                         stars,
                         ha='center', va='bottom',
                         fontsize=12)

        plt.xticks(np.arange(len(bin_keys)), bin_keys, rotation=45)
        plt.xlabel("Intensity Bins")
        plt.ylabel(rp_name)
        plt.title(f"Binned Analysis: {rp_name}")
        plt.legend()
        plt.tight_layout()

        if save_image and save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            filename = f"{rp_name}_plot_with_different_threshold.jpg"
            plt.savefig(os.path.join(save_path, filename), dpi=300)

        plt.show()
