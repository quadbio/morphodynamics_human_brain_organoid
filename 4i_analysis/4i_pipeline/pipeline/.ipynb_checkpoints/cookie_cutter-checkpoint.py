from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phenoscapes.utils import get_metadata, scale_image
from scipy import ndimage
from skimage import img_as_uint, io, measure
from skimage.filters import gaussian, threshold_otsu
from skimage.filters.rank import median
from skimage.morphology import disk


def mask_cookie_cutter(
    sample: str,
    dir_input: str,
    cycle: int = 1,
    channels: list = [0, 1, 2, 3],
    sigma: int = 100,
    n_binary: int = 50,
    outlier_threshold: int = 30,
    smooth_area: bool = True,
    plot: bool = True,
    config: dict = None,
):

    """Creating an initial simple mask

    :param config:
    :param sigma:
    :param n_binary:
    :param outlier_threshold:
    :param smooth_area:
    :param sample: (str) id of sample/region
    :param dir_input: (str) path to input directory
    :param dir_output: (str) path to output directory
    :param cycle: (str) cycle to create mask from
    :param channels: (list) channels to create mask from
    :param save: (bool) save created mask
    :param plot: (bool) create plot
    :param save_plot: (bool) save plot
    :param show_plot: (bool) return plot

    :return: depending on input parameters the function saves the created mask and/or plots and/or shows the overview plot
    """

    dir_images = Path(dir_input, sample)
    df = get_metadata(dir_images, custom_regex=config["regex"])
    # List images
    # filter cycle
    df = df[df["cycle_id"] == cycle]
    # filter channels
    df = df[df["channel_id"].isin(channels)]
    # list files
    files_images = df[df["cycle_id"] == cycle]["file"].values
    # Load image
    print("Loading images...")
    img_init = np.dstack(
        [io.imread(Path(dir_images, file_image)) for file_image in files_images]
    )
    print("Scaling images...")
    for i in range(img_init.shape[2]):
        img_init[..., i] = scale_image(img_init[..., i])
    print("Initial masking...")
    # max all channels
    img_init = np.max(img_init, axis=2)
    # Apply gaussian
    img = gaussian(img_init, sigma=sigma)

    # Otsu thresholding
    thr = threshold_otsu(img)
    img = (img > thr).astype(int)

    # Fill holes
    img = ndimage.binary_fill_holes(img).astype("uint8")

    # Median filter to remove outliers
    img = median(img, disk(outlier_threshold))

    if smooth_area:
        print("Final masking...")
        # Smoothing selected area
        # Dilation
        struct = ndimage.generate_binary_structure(2, 2)
        img = ndimage.morphology.binary_dilation(
            img, structure=struct, iterations=n_binary
        ).astype(int)
        # Apply gaussian
        img = gaussian(img, sigma=int(sigma / 2))
        # Apply erosion
        img = ndimage.morphology.binary_erosion(
            img, structure=struct, iterations=n_binary
        ).astype(int)
        # Otsu thresholding
        thr = threshold_otsu(img)
        img = (img > thr).astype(int)
        # Fill holes
        img = ndimage.binary_fill_holes(img).astype(int)

    return img_as_uint(img)
