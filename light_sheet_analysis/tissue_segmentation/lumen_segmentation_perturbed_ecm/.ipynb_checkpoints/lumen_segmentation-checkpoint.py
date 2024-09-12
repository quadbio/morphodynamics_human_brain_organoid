import os
from functools import partial

import h5py
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from absl import app, flags
from lumen_segmenter import (
    keep_largest_object,
    postprocess_masks,
    read_and_stack,
    smooth_organoids,
)
from scipy import ndimage as ndi
from skimage import exposure, feature, future
from skimage.io import imread, imsave
from skimage.transform import downscale_local_mean, rescale, resize
from sklearn.ensemble import RandomForestClassifier

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "file_path",
    "",
    "path to input file",
)

flags.DEFINE_string(
    "output_dir",
    "",
    "path to output dir",
)

flags.DEFINE_string("time_point", "", "time point")


def main(argv):
    t = int(FLAGS.time_point)
    file_path = FLAGS.file_path
    output_dir = FLAGS.output_dir
    # Create folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    downsample_xy = 0.25
    h5_file = True
    sigma_min = 1
    sigma_max = 256 * downsample_xy

    features_func = partial(
        feature.multiscale_basic_features,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    import joblib

    # Loacation of rf classifier
    rf_location = "/cluster/home/gutgi/git_repositories/morphodynamics-of-human-brain-organoid-patterning/light_sheet_analysis/tissue_segmentation/rf_classifier/"

    # load RF
    clf = joblib.load(rf_location + "lumen_agar_v15_05.joblib")
    if h5_file:
        stack_file = h5py.File(file_path, "r")
        stack_shape = stack_file[f"t{t:05}"]["s00"]["0"]["cells"].shape[0]

    if h5_file == False:
        stack_cherry = (
            imread(file_path + f"/mCherry/t{t:04}_mCherry.tif")
            .astype(np.float32)
            .copy()
        )
        stack_gfp = (
            imread(file_path + f"/GFP/t{t:04}_GFP.tif").astype(np.float32).copy()
        )
        stack_combined = stack_cherry + stack_gfp
        stack_shape = stack_combined.shape[0]

    # Pixel wise segmentation
    combined_masks_raw = []
    for i in range(stack_shape):
        if h5_file:
            stack_file = h5py.File(file_path, "r")
            stack_cherry = (
                stack_file[f"t{t:05}"]["s00"]["0"]["cells"][i].astype(np.float32).copy()
            )
            stack_gfp = (
                stack_file[f"t{t:05}"]["s01"]["0"]["cells"][i].astype(np.float32).copy()
            )
            segment = stack_cherry + stack_gfp
        else:
            segment = stack_combined[i].copy()
        segment = rescale(
            segment, [downsample_xy, downsample_xy], order=1, preserve_range=True
        )
        features = features_func(segment)
        result = future.predict_segmenter(features, clf)
        combined_masks_raw.append(result.astype(np.uint8))
    combined_masks_raw = np.array(combined_masks_raw)

    # Postprocessing
    footprint = skimage.morphology.disk(3)

    # Smoothing with a gaussian of sigm 2
    smooth_organoid = smooth_organoids(np.array(combined_masks_raw), 2)

    # 2D Postprocessing to remove unlikely masks
    combined_masks_processed = []
    for one_slice in smooth_organoid:
        lumen_closed = ~skimage.morphology.binary_closing(
            ~np.array(one_slice == 3), footprint
        )
        organoid_closed = skimage.morphology.binary_closing(
            np.array(one_slice == 2), footprint
        )

        combined_closed = np.ones(lumen_closed.shape)
        combined_closed[organoid_closed] = 2
        combined_closed[lumen_closed] = 3
        combined_masks_processed.append(postprocess_masks(combined_closed))
    combined_masks_processed = np.array(combined_masks_processed)
    # Only keep largest object (organoid)
    combined_masks_processed = keep_largest_object(combined_masks_processed)

    # Save both raw + processed masks
    imsave(
        output_dir + f"{t:04}" + "_lumen_organoid_mask_processed.tif",
        combined_masks_processed.astype(np.uint8),
        plugin="tifffile",
        check_contrast=False,
        compress=6,
        bigtiff=True,
    )

    imsave(
        output_dir + f"{t:04}" + "_lumen_organoid_mask_raw.tif",
        combined_masks_raw.astype(np.uint8),
        plugin="tifffile",
        check_contrast=False,
        compress=6,
        bigtiff=True,
    )


if __name__ == "__main__":
    app.run(main)
