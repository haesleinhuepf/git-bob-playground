---
name: BioImage Analysis
about: Submit a microscopy image for AI-assisted analysis
title: "[BioImage Analysis]: "
labels: image-analysis
---

## Analysis Goal (What should be done / analysed?)


## Image Upload
📎 **Drag & drop your microscopy image here** (JPG, PNG, GIF, e.g. 512x512 pixels, 2D only).

## Python Tools
- List of Python libraries we should use for answering this:
  - numpy
  - scikit-image
  - napari-simpleitk-image-processing
  - napari-segment-blobs-and-things-with-membranes
  - stackview
  - pandas
  - seaborn
  - scipy

**Note:** Your images and the text you enter here may be sent to [OpenAI](https://openai.com/)'s online service ([Third party terms of use](https://openai.com/policies/row-terms-of-use/)) or [Anthropic's claude](https://www.anthropic.com/api) online service ([Terms of service](https://www.anthropic.com/legal/consumer-terms)) or [Google AI](https://ai.google.dev/) ([Terms of service](https://ai.google.dev/gemini-api/terms)) where we use a large language model to answer your request. 
Do not upload any data you cannot share openly. Also do not enter any private or secret information. By submitting this Github issue, you confirm that you understand these conditions.

Once submitted, @haesleinhuepf will review and consult [git-bob, an AI-assistant](https://github.com/haesleinhuepf/git-bob) for bio-image analysis suggestions. 

<details>
    <summary>Detailed instructions for bio-image analysis using Python (feel free to modify)</summary>

## Detailed Python Bio-image Analysis instructions

If the following tasks are requested, we can adapt the code corresponding snippets:

### Viewing images using stackview

When you use stackview, you always start by importing the library: `import stackview`.
      
* Showing an image stored in variable `image` and a segmented image stored in variable `labels` on top with animated blending. Also works with two images or two label images.
  stackview.animate_curtain(image, labels)

* Showing an animation / timelapse image stored in variable `image`.
  stackview.animate(image)
  
* Save an animation / timelapse stored in variable `image` with specified frame delay to a file.
  stackview.animate(image, filename="output.gif", frame_delay_ms=100)
  
* Display an image stored in a variable `image` (this also works with label images). Prefer stackview.insight over matplotlib.pyplot.imshow!
  stackview.insight(image)

* Display an image as a label image explicitly.
  stackview.imshow(image, labels=True)

### Processing images using the napari-simpleitk-image-processing (nsitk) Python library. 

When you use nsitk, you always start by importing the library: `import napari_simpleitk_image_processing as nsitk`.
When asked for specific tasks, you can adapt one of the following code snippets:
  
* Apply a median filter to an image to remove noise while preserving edges.
  nsitk.median_filter(image, radius_x=2, radius_y=2)
  
* Applies Otsu's threshold selection method to an intensity image and returns a binary image (also works with intermodes, kittler_illingworth, li, moments, renyi_entropy, shanbhag, yen, isodata, triangle, huang and maximum_entropy instead of otsu).
  nsitk.threshold_otsu(image)
  
* Computes the signed Maurer distance map of the input image.
  nsitk.signed_maurer_distance_map(binary_image)
  
* Detects edges in the image using Canny edge detection.
  nsitk.canny_edge_detection(image, lower_threshold=0, upper_threshold=50)
  
* Identifies the regional maxima of an image.
  nsitk.regional_maxima(image)
  
* Rescales the intensity of an input image to a specified range.
  nsitk.rescale_intensity(image, output_minimum=0, output_maximum=255)
  
* Applies the Sobel operator to an image to find edges.
  nsitk.sobel(image)
  
* Enhances the contrast of an image using adaptive histogram equalization.
  nsitk.adaptive_histogram_equalization(image, alpha=0.3, beta=0.3, radius_x=5, radius_y=5)
  
* Applies a standard deviation filter to an image.
  nsitk.standard_deviation_filter(image, radius_x=5, radius_y=5)
  
* Labels the connected components in a binary image.
  nsitk.connected_component_labeling(binary_image)
  
* Labels objects in a binary image and can split object that are touching..
  nsitk.touching_objects_labeling(binary_image)

* Applies the Laplacian of Gaussian filter to find edges in an image.
  nsitk.laplacian_of_gaussian_filter(image, sigma=1.0)
  
* Identifies h-maxima of an image, suppressing maxima smaller than h.
  nsitk.h_maxima(image, height=10)
  
* Removes background in an image using the Top-Hat filter.
  nsitk.white_top_hat(image, radius_x=5, radius_y=5)

* Computes basic statistics for labeled object regions in an image.
  nsitk.label_statistics(image, label_image, size=True, intensity=True, shape=False)
  
* Computes a map from a label image where the pixel intensity corresponds to the number of pixels in the given labeled object (analogously work elongation_map, feret_diameter_map, roundness_map).
  nsitk.pixel_count_map(label_image)
  
### Processing images using napari-segment-blobs-and-things-with-membranes (nsbatwm)

If you use this plugin, you need to import it like this: `import napari_segment_blobs_and_things_with_membranes as nsbatwm`. 
You can then use it for various purposes:

* Denoise an image using a Gaussian filter
  nsbatwm.gaussian_blur(image, sigma=1)

* Denoise an image, while preserving edges:
  nsbatwm.median_filter(image, radius=2)

* Denoise an image using a percentile (similar to median, but free in choosing the percentile)
  nsbatwm.percentile_filter(image, percentile=50, radius=2)

* Determine the local minimum intensity for every pixel (also works with maximum)
  nsbatwm.minimum_filter(image, radius=2)

* Enhance edges
  nsbatwm.gaussian_laplace(image, sigma=2)

* Remove background from an image using the Top-Hat filter
  nsbatwm.white_tophat(image, radius=2)

* Remove background from an image using the Rolling-Ball method
  nsbatwm.subtract_background(membranes, rolling_ball_radius=15)

* Uses combination of Voronoi tesselation and Otsu's threshold method for segmenting an image
  nsbatwm.voronoi_otsu_labeling(blobs, spot_sigma=3.5, outline_sigma=1)

* Apply a Gaussian blur, Otsu's threshold for binarization and returns a label image
  nsbatwm.gauss_otsu_labeling(blobs, outline_sigma=1)

* Binarize an image using a threshold determined using Otsu's method (also works with li, triangle, yen, mean methods)
  nsbatwm.threshold_otsu(blobs)

* Split touching objects in a binary image
  nsbatwm.split_touching_objects(binary, sigma=3.5)

* Identify individual objects in a binary image using Connected Component labeling
  nsbatwm.connected_component_labeling(binary)

* Apply a Watershed algorithm to an an image showing membrane-like structures and a label image that serves as seeds for the watershed
  nsbatwm.seeded_watershed(membranes_image, labeled_seeds)

* Apply a Watershed algorithm to an image showing membrane-like structures. The seeds for the watershed are internally determined using local minima.
  nsbatwm.local_minima_seeded_watershed(membrane_image, spot_sigma=10, outline_sigma=0)

* Dilate labels to increase their size 
  nsbatwm.expand_labels(label_image, distance=1)

* Smooths outlines of label images by determining the most popular label locally
  nsbatwm.mode_filter(label_image, radius=10)

* Remove labels that touch the image border
  nsbatwm.remove_labels_on_edges(label_image)

* Skeletonize labels
  nsbatwm.skeletonize(labels)

### Working with Pandas DataFrames

In case a pandas DataFrame, e.g. `df` is the result of a code block, just write `df.head()`
by the end so that the user can see the intermediate result.

### Processing images with scikit-image (skimage)
  
* Load an image file from disc and store it in a variable:
  from skimage.io import imread
  image = imread(filename)

* Save an image file to disc:
  from skimage.io import imwrite
  imread(filename, image)
  
* Expanding labels by a given radius in a label image works like this:
  from skimage.segmentation import expand_labels
  expanded_labels = expand_labels(label_image, distance=10)
  
* Turn a label image into an RGB image, e.g. for saving as png:
  from skimage import color
  rgb_image = (color.label2rgb(labels, bg_label=0, kind='overlay')*255).astype('uint8')
  
* Measure properties of labels with respect to an image works like this:
  import pandas as pd
  from skimage.measure import regionprops_table
  properties = ['label', 'area', 'mean_intensity'] # add more properties if needed
  measurements = regionprops_table(label_image, intensity_image=image, properties=properties)
  df = pd.DataFrame(measurements)

</details>
