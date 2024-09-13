import re

import anndata
import h5py
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import psutil
import pymeshfix
import pyvista as pv
import scanpy as sc
import skimage
from IPython.display import HTML, clear_output
from morphometrics.measure import measure_selected
from PIL import Image
from skimage import draw
from skimage.io import imread, imsave
from skimage.measure import label, marching_cubes, regionprops, regionprops_table
from skimage.transform import downscale_local_mean, rescale, resize
from tqdm import tqdm


def read_extract(t):
    stack = imread(f"{input_dir}/images/{all_files[t]}")
    mask = imread(f"{input_dir}/predictions/{all_files[t]}")
    morpho_data = pd.read_csv(output_dir + all_files[t].replace(".tif", ".csv"))
    # Get only masks which were measured
    from_values = np.arange(np.max(mask) + 1)
    to_values = np.zeros(from_values.shape)
    to_values[morpho_data["label"]] = from_values[morpho_data["label"]]
    mask = extract_masks(mask, from_values, to_values)

    region_properties_table = regionprops_table(
        mask, intensity_image=stack, properties=("label", "bbox", "intensity_image")
    )
    region_properties_table = pd.DataFrame(region_properties_table)
    bbox_images = []
    for j in range(len(region_properties_table)):
        bbox_list = []
        for bbox in range(6):
            if bbox < 3:
                bbox_loc = region_properties_table.iloc[j][f"bbox-{str(bbox)}"] - 8
                if bbox_loc < 0:
                    bbox_list.append(
                        region_properties_table.iloc[j][f"bbox-{str(bbox)}"]
                    )
                else:
                    bbox_list.append(bbox_loc)
            else:
                bbox_loc = region_properties_table.iloc[j][f"bbox-{str(bbox)}"] + 8
                if bbox_loc < 0:
                    bbox_list.append(
                        region_properties_table.iloc[j][f"bbox-{str(bbox)}"]
                    )
                else:
                    bbox_list.append(bbox_loc)

        bbox_images.append(
            stack[
                bbox_list[0] : bbox_list[3],
                bbox_list[1] : bbox_list[4],
                bbox_list[2] : bbox_list[5],
            ]
        )

    assert (morpho_data["label"] == region_properties_table["label"]).all()
    morpho_data["intensity_image"] = region_properties_table["intensity_image"]
    morpho_data["bbox_image"] = bbox_images

    if "AGAR" in experiment_directory:
        morpho_data["time_point"] = int(re.findall(r"\d+", all_files[t])[0])
        morpho_data["position"] = all_files[t].split("_")[-1].split(".")[0]
        morpho_data["experiment"] = "AGAR"
    else:
        morpho_data["time_point"] = (
            (int(re.findall(r"\d+", all_files[t])[0]) - 1) / 2
        ) + 1
        morpho_data["position"] = "8"
        morpho_data["experiment"] = "multimosaic"

    if "mCherry" in all_files[t]:
        morpho_data["channel"] = "mCherry"
    elif "GFP" in all_files[t]:
        morpho_data["channel"] = "GFP"

    return morpho_data
