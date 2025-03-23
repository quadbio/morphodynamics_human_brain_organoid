import os
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

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", "", "Path to where the data is.")
flags.DEFINE_string("array_num", "", "Array number.")


def main(argv):
    input_path = FLAGS.input_path
    array_num = int(FLAGS.array_num)

    dir_stitched = Path(input_path, "stitched")
    well_id = os.listdir(dir_stitched)[array_num]
    print(f"running well {well_id}")
    process_sample(
        well_id,
        dir_input=input_path,
        dir_output=input_path,
        metadata_file=f"{input_path}/stain_metadata.csv",
        config_file=f"{input_path}/config.yaml",
    )


if __name__ == "__main__":
    app.run(main)
