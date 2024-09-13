# From lightinisght.utils import h5_to_omezarr()
import h5py
import numpy as np
import zarr
from absl import app, flags
from lightinsight.utils.dataconverter import tiff_to_omezarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
from skimage.transform import rescale
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("ome_tiff", "", "ome tiff folder")
flags.DEFINE_string("out_path", "", "output path of the omezarr file")


def main(argv):
    input_path = FLAGS.ome_tiff
    out_path = FLAGS.out_path
    tiff_to_omezarr(
        input_path,
        out_path,
        n_jobs=4,
        downscale=[1, 0.5, 0.5],
    )


if __name__ == "__main__":
    app.run(main)
