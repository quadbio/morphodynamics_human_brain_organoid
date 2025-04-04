import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from absl import app, flags
from cookie_cutter import mask_cookie_cutter
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from phenoscapes.cli import load_default_config, process_sample
from phenoscapes.utils import get_metadata
from skimage import measure
from tifffile import imread, imwrite
from tqdm import tqdm


def pad_to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(
        a,
        (
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (x_pad // 2, x_pad // 2 + x_pad % 2),
        ),
        mode="constant",
    )


def cut_and_save(image, region, output_folder_stitched, region_shape, dir_raw, well_id):
    image_np = imread(f"{dir_raw}/{well_id}/" + image)
    cutout_image = image_np[
        int(region["bbox-0"]) : int(region["bbox-2"]),
        int(region["bbox-1"]) : int(region["bbox-3"]),
    ]
    cutout_image = pad_to_shape(cutout_image, region_shape)
    imwrite(output_folder_stitched + image, cutout_image)


FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", "", "Path to where the data is.")
flags.DEFINE_string("segment_channel", "", "Channel to segment")
flags.DEFINE_string("segment_cycle", "", "Cycel to segment")
flags.DEFINE_string("pannel", "", "Stain pannel")
flags.DEFINE_string(
    "cutter_size", "128", "Size of the area around the cookie cutter object to cut"
)


def main(argv):
    input_path = FLAGS.input_path
    cutter_size = int(FLAGS.cutter_size)

    # set working directory
    os.chdir(input_path)
    dir_4i_retina_example = os.getcwd()

    regex_exp = r"Stitched_C(?P<cycle_id>\d{1,2})_R(?P<well_id>\d{2,3})_ch(?P<channel_id>\d{2})\.tif$"

    cut_off_size = 1000000
    # load default config file
    config = load_default_config()
    # add regex to config
    config["regex"] = regex_exp
    # add path to stain metadata to config
    config["stain_metadata"] = str(Path(dir_4i_retina_example, "stain_metadata.csv"))
    # set nuclei channel
    config["channel_nuclei"] = 0
    # set staining channels
    config["channel_stains"] = [1, 2, 3]
    # set stain nuclei
    config["stain_nuclei"] = "DAPI"
    # Set ref cycle
    config["reference_cycle"] = 0
    # Set alignemnt ref cycles
    config["alignment"]["reference_cycles"] = None
    # Set parameter maps
    config["alignment"]["param_maps"] = {
        "rigid": "/param_maps/translation.txt",
        "affine": "/param_maps/affine.txt",
    }
    # Cycle DAPI
    config["simple_masking"]["sigma"] = 50
    config["simple_masking"]["n_binary_operations"] = 50

    config["refined_masking"]["cycle_dapi"] = 0
    config["refined_masking"]["sigma"] = 50
    config["refined_masking"]["n_binary_operations"] = 30
    config["refined_masking"]["use_only_dapi"] = True
    config["cropping_denoising"]["n_processes"] = 3

    config["cellpose"]["segment_channel"] = int(FLAGS.segment_channel)
    config["cellpose"]["segment_cycle"] = int(FLAGS.segment_cycle)
    config["cellpose"]["diameter"] = 40.8
    config["cellpose"]["cellprob_threshold"] = -1.0
    config["cellpose"]["flow_threshold"] = 0.8
    config["analysis_pipeline"]["slice_step"] = 1000
    config["cropping_denoising"]["mask_zeros_smo"] = True
    config["analysis_pipeline"]["stat"] = "median"
    config["speckle_removal"]["percentile"] = 3
    config["feature_extraction"]["expand_labels"] = 0

    # Only run masking
    config["stitching"]["run"] = True
    config["simple_masking"]["run"] = True
    config["alignment"]["run"] = True
    config["refined_masking"]["run"] = True
    config["alignment_check"]["run"] = True
    config["cropping_denoising"]["run"] = True
    config["speckle_removal"]["run"] = True
    config["bg_subtraction"]["run"] = True
    config["cellpose"]["run"] = True
    config["feature_extraction"]["run"] = True
    config["analysis_pipeline"]["run"] = True
    config["analysis_pipeline"]["run"] = True
    config["feature_morphology_extraction"]["run"] = True
    # Cookie cutter!
    dir_input = input_path
    f"{input_path}/stain_metadata.csv"
    dir_stitched = Path(dir_input, "stitched")
    dir_raw = Path(dir_input, "raw")
    dirs = os.listdir(dir_raw)
    for well_id in os.listdir(dir_raw):
        input_path = Path(dir_raw, well_id)
        image_names = os.listdir(input_path)
        image_names = [image for image in image_names if ".tif" in image]
        try:
            for image_name in image_names:
                cycle, _, _, channel = re.search(
                    r"C(\d+)xy(\d+)__Channel_c(\d+)_stitch_ch(\d+).tif", image_name
                ).groups()
                new_name = f"Stitched_C{cycle}_{well_id}_ch{channel}.tif"
                os.rename(Path(input_path, image_name), Path(input_path, new_name))
        except:
            print(f"{well_id} is already correct")

    for well_id in os.listdir(dir_raw):
        input_path = Path(dir_raw, well_id)
        image_names = os.listdir(input_path)
        image_names = [image for image in image_names if ".tif" in image]
        for image_name in image_names:
            if image_name == f"Stitched_C05_{well_id}_chh_.tif" and not (
                f"Stitched_C05_{well_id}_ch00.tif" in image_names
            ):
                new_name = f"Stitched_C05_{well_id}_ch00.tif"
                os.rename(Path(input_path, image_name), Path(input_path, new_name))
            if image_name == f"Stitched_C04_{well_id}_chh_.tif" and not (
                f"Stitched_C04_{well_id}_ch00.tif" in image_names
            ):
                new_name = f"Stitched_C04_{well_id}_ch00.tif"
                os.rename(Path(input_path, image_name), Path(input_path, new_name))

    for well_id in dirs:
        print(well_id)
        initial_mask = mask_cookie_cutter(
            sample=f"{well_id}",
            cycle=config["reference_cycle"],
            channels=config["simple_masking"]["channels"],
            sigma=10,
            n_binary=5,
            dir_input=dir_raw,
            outlier_threshold=5,
            config=config,
        )
        initial_mask = measure.label(initial_mask)
        regions = pd.DataFrame(
            measure.regionprops_table(
                initial_mask, properties=("label", "bbox", "area")
            )
        )
        regions = regions[regions["area"] > cut_off_size]
        upper_lim = np.array(initial_mask.shape).max()
        regions["bbox-0"] = (regions["bbox-0"] - cutter_size).clip(
            lower=0, upper=upper_lim
        )
        regions["bbox-1"] = (regions["bbox-1"] - cutter_size).clip(
            lower=0, upper=upper_lim
        )
        regions["bbox-2"] = (regions["bbox-2"] + cutter_size).clip(
            lower=0, upper=upper_lim
        )
        regions["bbox-3"] = (regions["bbox-3"] + cutter_size).clip(
            lower=0, upper=upper_lim
        )

        images = os.listdir(f"{dir_raw}/{well_id}/")
        images = [image for image in images if ".tif" in image]

        for j in range(len(regions)):
            region = regions.iloc[j]
            output_folder_stitched = f"{dir_stitched}/{well_id}_{j}/"
            if not os.path.exists(output_folder_stitched):
                os.makedirs(output_folder_stitched)
            # Add region_shape --> pad image if necessary --> pad to shape -->
            region_shape = (
                int(region["bbox-2"] - region["bbox-0"]),
                int(region["bbox-3"] - region["bbox-1"]),
            )
            Parallel(n_jobs=4, verbose=2)(
                delayed(cut_and_save)(
                    image,
                    region,
                    output_folder_stitched,
                    region_shape,
                    dir_raw,
                    well_id,
                )
                for image in images
            )

    df = pd.DataFrame()
    dirs = os.listdir("stitched")
    # dirs=['294','309']
    for directory in dirs:
        df = pd.concat([df, get_metadata(f"stitched/{directory}", regex_exp)])

    pannel = pd.read_csv(f"{dir_input}/{FLAGS.pannel}", header=None)
    pannel = pannel.iloc[:, :4]

    pannel.columns = ["Cycle", "AB1", "AB2", "AB3"]
    pannel["AB0"] = "DAPI"
    names = []
    for i in range(len(df)):
        entry = df.iloc[i]
        channel = entry["channel_id"]
        cycle = entry["cycle_id"]
        AB_name = (
            pannel[pannel["Cycle"] == cycle][f"AB{channel}"]
            .iloc[0]
            .split(" ")[0]
            .replace("/", "")
        )
        names.append(AB_name)
    df["stain"] = names

    # generate stain metadata from file names
    df_stains = (
        df[["cycle_id", "channel_id", "stain"]]
        .drop_duplicates()
        .sort_values(by=["cycle_id", "channel_id"])
    )
    # remove 'excluded ' from stain names
    df_stains["stain"] = df_stains["stain"].apply(lambda x: x.replace("excluded ", ""))
    # where stain is elution or native set stain to NaN
    df_stains.loc[df_stains["stain"].isin(["elu", "none", "nan"]), "stain"] = np.nan
    # delete stain rows where stain is DAPI
    df_stains = df_stains[~df_stains["stain"].isin(["DAPI"])]
    # write stain metadata to csv
    df_stains.to_csv(Path(dir_4i_retina_example, "stain_metadata.csv"), index=False)

    # Wrtie config
    with open(Path(dir_4i_retina_example, "config.yaml"), "w") as file:
        yaml.dump(config, file, sort_keys=False)


if __name__ == "__main__":
    app.run(main)
