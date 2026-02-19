    import numpy as np
    import cupy as cp
    from tqdm import tqdm
    from tifffile import imwrite
    import os

def plot_spots_gpu(spots, image, spot_radius=4, save_plot=False, plotname='spotPlot.tif', save_path='./'):
    """
    Plot spots on the image using GPU acceleration via CuPy.
    The output array matches the dtype of the original image.
    """


    dtype = image.dtype

    # Create CuPy array for plotting
    spotPlot_gpu = cp.zeros(image.shape, dtype=dtype)

    for i in tqdm(range(len(spots)), desc="Plotting spots on GPU"):
        shape = [spotPlot_gpu.shape[1], spotPlot_gpu.shape[2]]
        y = spots[i][1]
        x = spots[i][2]
        z = spots[i][0]

        plot_location = generate_coordinates_2D(
            y, x, shape, iteration=spot_radius, get_inner_spot=False
        )

        for item in plot_location:
            if np.issubdtype(dtype, np.integer):
                spotPlot_gpu[z, item[0], item[1]] = np.iinfo(dtype).max
            else:
                spotPlot_gpu[z, item[0], item[1]] = 1.0

    # Transfer back to CPU for saving
    spotPlot = cp.asnumpy(spotPlot_gpu)

    if save_plot:
        imwrite(os.path.join(save_path, plotname), spotPlot, photometric='minisblack')
    
    return spotPlot

