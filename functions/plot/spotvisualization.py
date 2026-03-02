import numpy as np
import cupy as cp
from tqdm import tqdm
from tifffile import imwrite
import os
def generate_coordinates_2D(y, x, shape, iteration =4, get_inner_spot = False):
    # Initial collection with the starting coordinate (x, y)
    coordinates_collection = [(y, x)]
    max_y = (shape[0]-1)
    max_x = (shape[1]-1)

    for _ in range(iteration):
        # Copy the current collection to avoid modifying it while iterating
        current_coordinates = coordinates_collection.copy()

        # Loop through each coordinate in the current collection
        for coord in current_coordinates:
            # Extract x and y values from the coordinate
            cy, cx = coord

            # Add neighboring coordinates to the collection if within bounds
            if cx + 1 <= max_x:
                coordinates_collection.append((cy,cx+1))
            if 0 <= cx - 1:
                coordinates_collection.append((cy,cx-1))
            if cy + 1 < max_y:
                coordinates_collection.append((cy+1, cx))
            if 0 <= cy - 1:
                coordinates_collection.append((cy-1, cx))

    # Remove duplicates by converting the list to a set and then back to a list
    coordinates_collection = list(set(coordinates_collection))
    if get_inner_spot:
        return coordinates_collection
    else:
        coordinates_collection = [coord for coord in coordinates_collection if abs(coord[0] - y) + abs(coord[1] - x) == iteration]
        return coordinates_collection

def plot_spots(spots, image, spot_radius=4, save_plot=False, plotname='spotPlot.tif', save_path='./'):
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

