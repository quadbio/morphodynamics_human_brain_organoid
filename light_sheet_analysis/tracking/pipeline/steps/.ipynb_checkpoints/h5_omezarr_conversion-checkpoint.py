# From lightinisght.utils import h5_to_omezarr()
import h5py
import numpy as np
import zarr
from absl import app, flags
from lightinsight.utils.dataconverter import h5_to_omezarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
from skimage.transform import rescale
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("h5_file", "", "h5 file")
flags.DEFINE_string("out_path", "", "output path of the omezarr file")


def main(argv):
    h5_file = FLAGS.h5_file
    out_path = FLAGS.out_path
    channel_names = ["GFP"]
    h5_channel_names = ["s01"]
    h5_to_omezarr(
        h5_file,
        out_path,
        channel_names,
        h5_channel_names,
        time_points=None,
        n_jobs=12,
        downscale=[1, 0.5, 0.5],
    )


if __name__ == "__main__":
    app.run(main)
