__all__ = ['show_images_in_grid', 'convert_dict_to_text']  # Specifies the function to be publicly accessible from this module.

import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
from google.cloud import storage
import io
import concurrent.futures


def show_images_in_grid(image_paths, ncols, subplot_aspect_ratio=1, fig_width=10, max_workers=None):
    """
    Displays images in a grid layout using matplotlib. Handles local and GCS image paths with multithreading.

    Args:
        image_paths (list): A list of image file paths (local or GCS paths)
        ncols (int): Number of columns in the grid
        subplot_aspect_ratio (float): Desired aspect ratio (width/height) for each subplot
        fig_width (int): Desired width of the overall figure in inches
        max_workers (int, optional): Maximum number of threads to use for parallel downloads. Defaults to None (uses the number of available CPUs).
    """

    num_images = len(image_paths)
    nrows = math.ceil(num_images / ncols)

    fig_height = fig_width * (nrows / ncols) * subplot_aspect_ratio
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nrows, ncols)

    storage_client = storage.Client()  # Initialize GCS client

    def load_image(image_path):
        if image_path.startswith("gs://"):
            bucket_name, blob_name = image_path[5:].split("/", 1)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            with io.BytesIO(blob.download_as_bytes()) as image_file:
                return plt.imread(image_file, format='png')
        else:
            return plt.imread(image_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit image loading tasks to the executor and store the futures
        futures = [executor.submit(load_image, path) for path in image_paths]

        # Iterate over the futures and display the images as they become available
        for i, future in enumerate(futures):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            img = future.result()  # Get the image from the completed future
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)

    plt.tight_layout()
    plt.show()
    
def convert_dict_to_text(input_dict):
    final_str = ""
    for value in input_dict.values():
        final_str += str(value)
    return final_str