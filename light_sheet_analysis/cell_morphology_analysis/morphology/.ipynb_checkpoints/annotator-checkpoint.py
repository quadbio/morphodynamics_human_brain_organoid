import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["figure.dpi"] = 300
from IPython.display import HTML, clear_output
from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import downscale_local_mean, rescale, resize

plt.style.use("dark_background")
import anndata
import h5py
import joblib
import napari
import numpy as np
import pandas as pd
import psutil
import pymeshfix
import pyvista as pv
import scanpy as sc
import skimage
from morphometrics.measure import measure_selected
from skimage import draw
from skimage.measure import label, marching_cubes, regionprops, regionprops_table
from tqdm import tqdm


def plot_save(labeled_DF, i, label_dir):
    # for i in tqdm(range(5)):
    fig, ax = plt.subplots(ncols=3, nrows=2, num=1, clear=True, figsize=(10, 5))
    image = labeled_DF.iloc[i]["intensity_image"]
    channel = labeled_DF.iloc[i]["channel"]
    experiment = labeled_DF.iloc[i]["experiment"]
    max_intensity = image.max()
    image = rescale(image, [2 / (0.347 * 2), 1, 1], anti_aliasing=False)
    vmax_image = np.percentile(image, 99.9)
    ax[0, 0].imshow(image.max(axis=0), cmap="gray", vmin=0, vmax=vmax_image)
    ax[0, 0].set_title(
        f"XY of channel {channel},{experiment}, \n max intensity of: {str(max_intensity)}",
        fontsize=8,
    )
    # ax[0].axis("off")
    ax[0, 1].imshow(image.max(axis=1), cmap="gray", vmin=0, vmax=vmax_image)
    ax[0, 1].set_title(f"XZ of channel {channel},{experiment}", fontsize=8)
    # ax[0,1].axis("off")
    ax[0, 2].imshow(image.max(axis=2), cmap="gray", vmin=0, vmax=vmax_image)
    ax[0, 2].set_title(f"YZ of channel {channel},{experiment}", fontsize=8)

    try:
        image = labeled_DF.iloc[i]["bbox_image"]
        channel = labeled_DF.iloc[i]["channel"]
        max_intensity = image.max()
        image = rescale(image, [2 / (0.347 * 2), 1, 1], anti_aliasing=False)
        vmax_bbox = np.percentile(image, 99.9)
        ax[1, 0].imshow(image.max(axis=0), cmap="gray", vmin=0, vmax=vmax_bbox)
        ax[1, 0].set_title(
            f"XY of channel {channel},{experiment}, \n max intensity of: {str(max_intensity)}",
            fontsize=8,
        )
        # ax[0].axis("off")
        ax[1, 1].imshow(image.max(axis=1), cmap="gray", vmin=0, vmax=vmax_bbox)
        ax[1, 1].set_title(f"XZ of channel {channel},{experiment}", fontsize=8)
        # ax[0,1].axis("off")
        ax[1, 2].imshow(image.max(axis=2), cmap="gray", vmin=0, vmax=vmax_bbox)
        ax[1, 2].set_title(f"YZ of channel {channel}, {experiment}", fontsize=8)
    except:
        print("not possible")
    # ax[2].axis("off")
    fig.tight_layout(pad=2.0)
    label = labeled_DF.iloc[i]["label"]
    experiment = labeled_DF.iloc[i]["experiment"]
    channel = labeled_DF.iloc[i]["channel"]
    position = labeled_DF.iloc[i]["position"]
    time_point = labeled_DF.iloc[i]["time_point"]
    plt.savefig(
        label_dir
        + f"structure_{experiment}_{channel}_{position}_{time_point}_{label}.jpg"
    )
    # plt.close(fig)


# Label images
def create_labels(image_path, file_name):

    labels = []
    image_path = image_path + "/"
    val_for = ["jpg"]
    images = os.listdir(image_path)
    images = [i for i in images if i.split(".")[-1] in val_for]

    for i in tqdm(images):
        im = Image.open(image_path + i)
        plt.imshow(im)
        plt.show()
        while True:
            try:
                inp = input()
                labels.append(int(inp))
                break
            except ValueError:
                print("Not a number, please try again...")
        clear_output(wait=True)

        data = {"Image": images, "Label": labels}

    label_data_frame = pd.DataFrame(data)
    if os.path.isfile(image_path + "/" + file_name):
        print("Annotations alread avalible, saving as _v2")
        label_data_frame.to_csv(
            (image_path + "/" + file_name).replace(".csv", "_v2.csv")
        )
    else:
        label_data_frame.to_csv(image_path + "/" + file_name)

    print("\nlabels.csv saved!")
    return label_data_frame


def extract_masks(mask, from_values, to_values):
    array = mask.astype(np.float32)
    sort_idx = np.argsort(from_values)
    idx = np.searchsorted(from_values, array, sorter=sort_idx)
    return (to_values[sort_idx][idx]).astype(np.uint16)
