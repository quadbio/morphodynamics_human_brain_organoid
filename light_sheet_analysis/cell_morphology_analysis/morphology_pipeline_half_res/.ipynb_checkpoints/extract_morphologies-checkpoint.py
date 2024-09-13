import glob
import os

import anndata
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import skimage
from absl import app, flags
from IPython.display import HTML, clear_output
from morphometrics.measure import measure_selected
from PIL import Image
from skimage.io import imread, imsave
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import downscale_local_mean, rescale, resize
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "file_path",
    "",
    "path to input file",
)

flags.DEFINE_string("time_point", "", "time point")


def main(argv):
    input_dir = FLAGS.file_path
    output_dir = input_dir + "/morphometrics/"
    t = int(FLAGS.time_point) - 1

    all_files = [os.path.basename(x) for x in glob.glob(f"{input_dir}/images/*")]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stack = imread(f"{input_dir}/images/{all_files[t]}")
    stack = rescale(
        stack, [2.8818443804, 1, 1], anti_aliasing=False, preserve_range=True
    )

    mask = imread(f"{input_dir}/predictions/{all_files[t]}")
    mask = rescale(
        mask, [2.8818443804, 1, 1], order=0, anti_aliasing=False, preserve_range=True
    ).astype(np.uint16)

    # Remove very small masks
    mask = skimage.morphology.remove_small_objects(mask, 100)

    major_axis = []
    minor_axis = []

    for region in tqdm(regionprops(mask)):
        major_axis.append(region.axis_major_length)
        try:
            minor_axis.append(region.axis_minor_length)
        except:
            minor_axis.append(np.nan)

    region_properties_table["time"] = str(t)

    region_properties_table = regionprops_table(
        mask,
        intensity_image=stack,
        properties=("label", "bbox", "intensity_image", "area"),
    )

    measurement_selection = [
        "surface_properties_from_labels",
        {
            "name": "regionprops",
            "choices": {
                "size": True,
                "intensity": True,
                "position": True,
                "moments": True,
            },
        },
    ]

    all_measurements = measure_selected(
        label_image=mask,
        intensity_image=stack,
        measurement_selection=measurement_selection,
    )

    all_measurements["axis_minor_length"] = minor_axis
    all_measurements["axis_major_length"] = major_axis
    all_measurements.to_csv(output_dir + all_files[t].replace(".tif", ".csv"))


if __name__ == "__main__":
    app.run(main)
