# -*- coding: utf-8 -*-
"""
A compact Python translation of selected Tier1 kernels using NumPy and scikit-image.

This module provides a subset of the long_file.java API for typical image processing
tasks on 2D and 3D images. Shapes follow the scikit-image convention:
2D images: (y, x), 3D images: (z, y, x).
"""
from __future__ import annotations

import numpy as np
from skimage import filters, morphology, util


def _as_bool(arr: np.ndarray) -> np.ndarray:
    """Convert image to boolean mask (nonzero -> True)."""
    return np.asarray(arr) != 0


def absolute(image: np.ndarray) -> np.ndarray:
    """
    Absolute value per pixel.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        |image|.
    """
    return np.abs(image)


def add_images_weighted(image0: np.ndarray, image1: np.ndarray, factor0: float = 1.0, factor1: float = 1.0) -> np.ndarray:
    """
    Weighted sum of two images.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.
    factor0 : float, optional
        Weight for first image, by default 1.0.
    factor1 : float, optional
        Weight for second image, by default 1.0.

    Returns
    -------
    ndarray
        factor0 * image0 + factor1 * image1.
    """
    return factor0 * np.asarray(image0) + factor1 * np.asarray(image1)


def add_image_and_scalar(image: np.ndarray, scalar: float = 1.0) -> np.ndarray:
    """
    Add a scalar to an image.

    Parameters
    ----------
    image : ndarray
        Input image.
    scalar : float, optional
        Scalar to add, by default 1.0.

    Returns
    -------
    ndarray
        image + scalar.
    """
    return np.asarray(image) + scalar


def binary_and(mask0: np.ndarray, mask1: np.ndarray) -> np.ndarray:
    """
    Logical AND of two binary images.

    Parameters
    ----------
    mask0 : ndarray
        First binary image (non-zero = True).
    mask1 : ndarray
        Second binary image (non-zero = True).

    Returns
    -------
    ndarray
        Resulting binary image (dtype=uint8, values {0,1}).
    """
    return np.logical_and(_as_bool(mask0), _as_bool(mask1)).astype(np.uint8)


def binary_or(mask0: np.ndarray, mask1: np.ndarray) -> np.ndarray:
    """
    Logical OR of two binary images.

    Parameters
    ----------
    mask0 : ndarray
        First binary image.
    mask1 : ndarray
        Second binary image.

    Returns
    -------
    ndarray
        Resulting binary image (dtype=uint8, values {0,1}).
    """
    return np.logical_or(_as_bool(mask0), _as_bool(mask1)).astype(np.uint8)


def binary_not(mask: np.ndarray) -> np.ndarray:
    """
    Logical NOT of a binary image.

    Parameters
    ----------
    mask : ndarray
        Binary image (non-zero = True).

    Returns
    -------
    ndarray
        Inverted binary image (dtype=uint8, values {0,1}).
    """
    return np.logical_not(_as_bool(mask)).astype(np.uint8)


def gaussian_blur(image: np.ndarray, sigma_x: float = 0.0, sigma_y: float = 0.0, sigma_z: float = 0.0,
                  preserve_range: bool = True) -> np.ndarray:
    """
    Gaussian blur with potentially anisotropic sigma.

    Parameters
    ----------
    image : ndarray
        Input image, 2D (y, x) or 3D (z, y, x).
    sigma_x : float, optional
        Sigma along x, by default 0.0.
    sigma_y : float, optional
        Sigma along y, by default 0.0.
    sigma_z : float, optional
        Sigma along z, by default 0.0 (ignored for 2D).
    preserve_range : bool, optional
        Keep original intensity range, by default True.

    Returns
    -------
    ndarray
        Blurred image.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        sigma = (sigma_y, sigma_x)
    elif img.ndim == 3:
        sigma = (sigma_z, sigma_y, sigma_x)
    else:
        raise ValueError("gaussian_blur expects 2D or 3D images")
    return filters.gaussian(img, sigma=sigma, preserve_range=preserve_range)


def convolve(image: np.ndarray, kernel: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """
    N-dimensional convolution.

    Parameters
    ----------
    image : ndarray
        Input image.
    kernel : ndarray
        Convolution kernel, same dimensionality as image.
    mode : {'reflect','constant','nearest','mirror','wrap'}, optional
        Border mode, by default 'reflect'.

    Returns
    -------
    ndarray
        Convolved image.
    """
    # use skimage.util.apply_parallel? keep simple with scipy if available; fall back to correlate
    try:
        from scipy.ndimage import convolve as ndi_convolve
        return ndi_convolve(np.asarray(image), np.asarray(kernel), mode=mode)
    except Exception:
        # correlate is convolution with flipped kernel; fine for symmetric kernels
        return filters.correlate(np.asarray(image), np.asarray(kernel), mode=mode)


def copy_image(image: np.ndarray) -> np.ndarray:
    """
    Copy image.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        A copy of the input image.
    """
    return np.array(image, copy=True)


def crop(image: np.ndarray, start_x: int, start_y: int, start_z: int = 0,
         width: int = 1, height: int = 1, depth: int = 1) -> np.ndarray:
    """
    Crop a sub-volume/region.

    Parameters
    ----------
    image : ndarray
        2D (y, x) or 3D (z, y, x) image.
    start_x : int
        Start x index.
    start_y : int
        Start y index.
    start_z : int, optional
        Start z index (3D only), by default 0.
    width : int, optional
        Width (x-size), by default 1.
    height : int, optional
        Height (y-size), by default 1.
    depth : int, optional
        Depth (z-size, 3D only), by default 1.

    Returns
    -------
    ndarray
        Cropped image.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        ys, xs = start_y, start_x
        return img[ys:ys + height, xs:xs + width]
    elif img.ndim == 3:
        zs, ys, xs = start_z, start_y, start_x
        return img[zs:zs + depth, ys:ys + height, xs:xs + width]
    else:
        raise ValueError("crop expects 2D or 3D images")


def gradient_x(image: np.ndarray) -> np.ndarray:
    """
    Gradient along x.

    Parameters
    ----------
    image : ndarray
        2D or 3D image.

    Returns
    -------
    ndarray
        dI/dx with same shape as image.
    """
    grads = np.gradient(np.asarray(image).astype(float))
    # order: (y, x) -> grads[1]; (z, y, x) -> grads[2]
    return grads[-1]


def gradient_y(image: np.ndarray) -> np.ndarray:
    """
    Gradient along y.

    Parameters
    ----------
    image : ndarray
        2D or 3D image.

    Returns
    -------
    ndarray
        dI/dy with same shape as image.
    """
    grads = np.gradient(np.asarray(image).astype(float))
    # order: (y, x) -> grads[0]; (z, y, x) -> grads[-2]
    return grads[-2]


def gradient_z(image: np.ndarray) -> np.ndarray:
    """
    Gradient along z.

    Parameters
    ----------
    image : ndarray
        3D image (z, y, x).

    Returns
    -------
    ndarray
        dI/dz with same shape as image.
    """
    img = np.asarray(image).astype(float)
    if img.ndim != 3:
        raise ValueError("gradient_z expects a 3D image")
    grads = np.gradient(img)
    return grads[0]


def dilate(binary: np.ndarray, connectivity: str = "box") -> np.ndarray:
    """
    Binary dilation with 'box' (Moore) or 'sphere' (von Neumann) neighborhood.

    Parameters
    ----------
    binary : ndarray
        Binary image (non-zero = True).
    connectivity : {'box', 'sphere'}, optional
        Structuring element shape, by default 'box'.

    Returns
    -------
    ndarray
        Dilated binary image (uint8).
    """
    mask = _as_bool(binary)
    if mask.ndim == 2:
        selem = morphology.square(3) if connectivity == "box" else morphology.disk(1)
    else:
        selem = morphology.cube(3) if connectivity == "box" else morphology.ball(1)
    out = morphology.binary_dilation(mask, footprint=selem)
    return out.astype(np.uint8)


def erode(binary: np.ndarray, connectivity: str = "box") -> np.ndarray:
    """
    Binary erosion with 'box' (Moore) or 'sphere' (von Neumann) neighborhood.

    Parameters
    ----------
    binary : ndarray
        Binary image (non-zero = True).
    connectivity : {'box', 'sphere'}, optional
        Structuring element shape, by default 'box'.

    Returns
    -------
    ndarray
        Eroded binary image (uint8).
    """
    mask = _as_bool(binary)
    if mask.ndim == 2:
        selem = morphology.square(3) if connectivity == "box" else morphology.disk(1)
    else:
        selem = morphology.cube(3) if connectivity == "box" else morphology.ball(1)
    out = morphology.binary_erosion(mask, footprint=selem)
    return out.astype(np.uint8)


def maximum_images(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Per-pixel maximum of two images.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        max(image0, image1).
    """
    return np.maximum(np.asarray(image0), np.asarray(image1))


def minimum_images(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Per-pixel minimum of two images.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        min(image0, image1).
    """
    return np.minimum(np.asarray(image0), np.asarray(image1))


def maximum_image_and_scalar(image: np.ndarray, scalar: float = 0.0) -> np.ndarray:
    """
    Per-pixel maximum of image and scalar.

    Parameters
    ----------
    image : ndarray
        Input image.
    scalar : float, optional
        Scalar, by default 0.0.

    Returns
    -------
    ndarray
        max(image, scalar).
    """
    return np.maximum(np.asarray(image), scalar)


def minimum_image_and_scalar(image: np.ndarray, scalar: float = 0.0) -> np.ndarray:
    """
    Per-pixel minimum of image and scalar.

    Parameters
    ----------
    image : ndarray
        Input image.
    scalar : float, optional
        Scalar, by default 0.0.

    Returns
    -------
    ndarray
        min(image, scalar).
    """
    return np.minimum(np.asarray(image), scalar)


def mask(image: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
    """
    Apply binary mask to an image.

    Parameters
    ----------
    image : ndarray
        Input intensity image.
    mask_image : ndarray
        Binary mask image (non-zero = True).

    Returns
    -------
    ndarray
        image where mask!=0, else 0.
    """
    img = np.asarray(image)
    m = _as_bool(mask_image)
    return np.where(m, img, np.zeros(1, dtype=img.dtype))


def equal(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Equality comparison per pixel.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        Binary result (uint8, values {0,1}).
    """
    return (np.asarray(image0) == np.asarray(image1)).astype(np.uint8)


def not_equal(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Non-equality comparison per pixel.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        Binary result (uint8, values {0,1}).
    """
    return (np.asarray(image0) != np.asarray(image1)).astype(np.uint8)


def greater(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Greater-than comparison per pixel.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        Binary result (uint8, values {0,1}).
    """
    return (np.asarray(image0) > np.asarray(image1)).astype(np.uint8)


def greater_or_equal(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Greater-or-equal comparison per pixel.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        Binary result (uint8, values {0,1}).
    """
    return (np.asarray(image0) >= np.asarray(image1)).astype(np.uint8)


def smaller(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Smaller-than comparison per pixel.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        Binary result (uint8, values {0,1}).
    """
    return (np.asarray(image0) < np.asarray(image1)).astype(np.uint8)


def smaller_or_equal(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Smaller-or-equal comparison per pixel.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        Binary result (uint8, values {0,1}).
    """
    return (np.asarray(image0) <= np.asarray(image1)).astype(np.uint8)


def logarithm(image: np.ndarray) -> np.ndarray:
    """
    Natural logarithm per pixel.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        log(image).
    """
    return np.log(np.asarray(image))


def exponential(image: np.ndarray) -> np.ndarray:
    """
    Base-e exponential per pixel.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        exp(image).
    """
    return np.exp(np.asarray(image))


def power(image: np.ndarray, scalar: float = 1.0) -> np.ndarray:
    """
    Raise pixels to a power.

    Parameters
    ----------
    image : ndarray
        Input image.
    scalar : float, optional
        Exponent, by default 1.0.

    Returns
    -------
    ndarray
        image ** scalar.
    """
    return np.power(np.asarray(image), scalar)


def square_root(image: np.ndarray) -> np.ndarray:
    """
    Square root per pixel.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        sqrt(image).
    """
    return np.sqrt(np.asarray(image))


def reciprocal(image: np.ndarray) -> np.ndarray:
    """
    Reciprocal 1/x per pixel.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        1.0 / image.
    """
    return 1.0 / np.asarray(image)


def multiply_images(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Per-pixel product of two images.

    Parameters
    ----------
    image0 : ndarray
        First image.
    image1 : ndarray
        Second image.

    Returns
    -------
    ndarray
        image0 * image1.
    """
    return np.asarray(image0) * np.asarray(image1)


def multiply_image_and_scalar(image: np.ndarray, scalar: float) -> np.ndarray:
    """
    Multiply image by scalar.

    Parameters
    ----------
    image : ndarray
        Input image.
    scalar : float
        Scalar.

    Returns
    -------
    ndarray
        image * scalar.
    """
    return np.asarray(image) * scalar


def divide_images(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    """
    Per-pixel division image0 / image1.

    Parameters
    ----------
    image0 : ndarray
        Numerator image.
    image1 : ndarray
        Denominator image.

    Returns
    -------
    ndarray
        image0 / image1.
    """
    return np.asarray(image0) / np.asarray(image1)


def divide_scalar_by_image(image: np.ndarray, scalar: float = 0.0) -> np.ndarray:
    """
    Divide scalar by image per pixel.

    Parameters
    ----------
    image : ndarray
        Denominator image.
    scalar : float, optional
        Numerator scalar, by default 0.0.

    Returns
    -------
    ndarray
        scalar / image.
    """
    return scalar / np.asarray(image)


def subtract_image_from_scalar(image: np.ndarray, scalar: float = 0.0) -> np.ndarray:
    """
    Compute scalar - image per pixel.

    Parameters
    ----------
    image : ndarray
        Input image.
    scalar : float, optional
        Scalar minuend, by default 0.0.

    Returns
    -------
    ndarray
        scalar - image.
    """
    return scalar - np.asarray(image)


def sobel_filter(image: np.ndarray) -> np.ndarray:
    """
    Sobel edge magnitude.

    Parameters
    ----------
    image : ndarray
        2D or 3D image.

    Returns
    -------
    ndarray
        Sobel filtered image.
    """
    return filters.sobel(np.asarray(image))


def maximum_x_projection(image: np.ndarray) -> np.ndarray:
    """
    Maximum projection along X (last axis).

    Parameters
    ----------
    image : ndarray
        2D/3D image.

    Returns
    -------
    ndarray
        Projection image.
    """
    return np.max(np.asarray(image), axis=-1)


def maximum_y_projection(image: np.ndarray) -> np.ndarray:
    """
    Maximum projection along Y (second-to-last axis in 2D/3D).

    Parameters
    ----------
    image : ndarray
        2D/3D image.

    Returns
    -------
    ndarray
        Projection image.
    """
    img = np.asarray(image)
    axis = -2 if img.ndim >= 2 else 0
    return np.max(img, axis=axis)


def maximum_z_projection(image: np.ndarray) -> np.ndarray:
    """
    Maximum projection along Z (axis 0 for 3D). For 2D, returns the image itself.

    Parameters
    ----------
    image : ndarray
        2D/3D image.

    Returns
    -------
    ndarray
        Projection image.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        return np.max(img, axis=0)
    return img.copy()


def transpose_xy(image: np.ndarray) -> np.ndarray:
    """
    Swap X and Y axes.

    Parameters
    ----------
    image : ndarray
        2D (y, x) or 3D (z, y, x).

    Returns
    -------
    ndarray
        Transposed image: (x, y) for 2D; (z, x, y) for 3D.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        return img.swapaxes(0, 1)
    if img.ndim == 3:
        return img.swapaxes(1, 2)
    raise ValueError("transpose_xy expects 2D or 3D images")


def transpose_xz(image: np.ndarray) -> np.ndarray:
    """
    Swap X and Z axes (3D only).

    Parameters
    ----------
    image : ndarray
        3D (z, y, x) image.

    Returns
    -------
    ndarray
        Transposed image (x, y, z).
    """
    img = np.asarray(image)
    if img.ndim != 3:
        raise ValueError("transpose_xz expects a 3D image")
    return img.swapaxes(0, 2)


def transpose_yz(image: np.ndarray) -> np.ndarray:
    """
    Swap Y and Z axes (3D only).

    Parameters
    ----------
    image : ndarray
        3D (z, y, x) image.

    Returns
    -------
    ndarray
        Transposed image (y, z, x).
    """
    img = np.asarray(image)
    if img.ndim != 3:
        raise ValueError("transpose_yz expects a 3D image")
    return img.swapaxes(0, 1)


def set_ramp_x(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """
    Create an image where pixels equal their X coordinate.

    Parameters
    ----------
    shape : tuple of int
        Output image shape (2D: (y, x), 3D: (z, y, x)).
    dtype : data-type, optional
        Output dtype, by default float32.

    Returns
    -------
    ndarray
        Ramp image along X.
    """
    if len(shape) == 2:
        y, x = shape
        return np.tile(np.arange(x, dtype=dtype), (y, 1))
    elif len(shape) == 3:
        z, y, x = shape
        ramp2d = np.tile(np.arange(x, dtype=dtype), (y, 1))
        return np.tile(ramp2d[None, ...], (z, 1, 1))
    else:
        raise ValueError("set_ramp_x expects 2D or 3D shape")


def set_ramp_y(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """
    Create an image where pixels equal their Y coordinate.

    Parameters
    ----------
    shape : tuple of int
        Output image shape (2D: (y, x), 3D: (z, y, x)).
    dtype : data-type, optional
        Output dtype, by default float32.

    Returns
    -------
    ndarray
        Ramp image along Y.
    """
    if len(shape) == 2:
        y, x = shape
        return np.tile(np.arange(y, dtype=dtype)[:, None], (1, x))
    elif len(shape) == 3:
        z, y, x = shape
        ramp2d = np.tile(np.arange(y, dtype=dtype)[:, None], (1, x))
        return np.tile(ramp2d[None, ...], (z, 1, 1))
    else:
        raise ValueError("set_ramp_y expects 2D or 3D shape")


def set_ramp_z(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """
    Create a 3D image where pixels equal their Z coordinate; for 2D returns zeros.

    Parameters
    ----------
    shape : tuple of int
        Output image shape (2D: (y, x), 3D: (z, y, x)).
    dtype : data-type, optional
        Output dtype, by default float32.

    Returns
    -------
    ndarray
        Ramp image along Z (or zeros for 2D).
    """
    if len(shape) == 2:
        return np.zeros(shape, dtype=dtype)
    elif len(shape) == 3:
        z, y, x = shape
        ramp = np.arange(z, dtype=dtype)[:, None, None]
        return np.tile(ramp, (1, y, x))
    else:
        raise ValueError("set_ramp_z expects 2D or 3D shape")
