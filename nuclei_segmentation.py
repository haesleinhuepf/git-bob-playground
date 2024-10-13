import numpy as np
from skimage.io import imread, imsave
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import stackview
from skimage import color

def segment_nuclei(image_url: str) -> np.ndarray:
    """
    Segments and labels bright nuclei in an image using the Voronoi-Otsu-Labeling algorithm.

    Parameters
    ----------
    image_url : str
        URL to the image to be processed.

    Returns
    -------
    np.ndarray
        Labelled image with segmented nuclei.
    """
    # Load the image
    image = imread(image_url, plugin='imageio')

    # Apply Voronoi-Otsu-Labeling
    labels = nsbatwm.voronoi_otsu_labeling(image, spot_sigma=3.5, outline_sigma=1)

    # Visualize the result
    stackview.imshow(image)
    stackview.imshow(labels, labels=True)

    # Convert label image to RGB and save as PNG
    rgb_image = color.label2rgb(labels, bg_label=0).astype("uint8")
    imsave("segmented_labels.png", rgb_image)

    return labels

# Example usage
if __name__ == "__main__":
    url = "https://github.com/user-attachments/assets/c1bf94c5-fe80-4ff9-a46e-d4d9cb9f276f"
    segment_nuclei(url)
