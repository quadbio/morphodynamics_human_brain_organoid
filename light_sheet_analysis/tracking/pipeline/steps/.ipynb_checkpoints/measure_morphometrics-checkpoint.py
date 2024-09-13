import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from absl import app, flags
from lightinsight.morphometrics.morphometrics import extract_morphometrics
from tqdm import tqdm

warnings.filterwarnings("ignore")
FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", "", "Input path of the omezarr file")
flags.DEFINE_string("morpho_path", "", "Path to save the morphometrics tables")
flags.DEFINE_string("array_section", "", "Section to segment")
flags.DEFINE_string("label_name", "", "Name of the used labels")
flags.DEFINE_string("n_jobs", "", "Number of jobs to start")
flags.DEFINE_string("zarr_level", "", "Number of jobs to start")


def main(argv):
    zarr_path = FLAGS.input_path
    array_section = FLAGS.array_section
    output_dir = FLAGS.morpho_path
    label_name = FLAGS.label_name
    n_jobs = int(FLAGS.n_jobs)
    zarr_level = FLAGS.zarr_level
    zarr_array = zarr.open(zarr_path, mode="r")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    channels = list(zarr_array.group_keys())
    for channel_name in channels:
        channel = zarr_array[channel_name]
        time_points_channel = sorted(np.array(list(channel.group_keys())).astype(int))
        time_points_labels = [
            t
            for t in time_points_channel
            if "labels" in list(channel[str(t)].group_keys())
        ]

        # Create list of all time points containing the label to be tracked
        time_points = [
            t
            for t in time_points_labels
            if label_name in list(channel[str(t)]["labels"].group_keys())
        ]

        time_points = np.array_split(np.array(time_points), 6)[int(array_section)]

        for time_point in time_points:
            extract_morphometrics(
                channel,
                time_point,
                channel_name,
                output_dir,
                label_name,
                n_jobs,
                zarr_level,
            )


if __name__ == "__main__":
    app.run(main)
