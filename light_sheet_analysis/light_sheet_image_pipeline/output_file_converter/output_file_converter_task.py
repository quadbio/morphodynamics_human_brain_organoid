import os

import npy2bdv
import numpy as np
from absl import app, flags
from skimage.io import imread, imsave

FLAGS = flags.FLAGS

flags.DEFINE_string("input_folder", "", "Input folder with the TIFF stacks")

flags.DEFINE_string("output_folder", "", "Output folder")

flags.DEFINE_string(
    "config_file",
    "/cluster/home/gutgi/light_sheet_image_pipeline/pipeline_config.yaml",
    "configuration file to use",
)

flags.DEFINE_string("f", "", "kernel")

flags.DEFINE_string("jobname", "test", "jobname")

flags.DEFINE_string("wait", "", "wait_argument")


def main(argv):
    FLAGS.config_file
    input_dir = FLAGS.input_folder
    output_dir = FLAGS.output_folder
    name_folder = input_dir.split("/")[-2]
    fname = output_dir + name_folder + "_processed.h5"

    bdv_writer = npy2bdv.BdvWriter(
        fname, nchannels=2, compression="gzip", overwrite=True
    )
    channels = ("mCherry", "GFP")
    bdv_writer.set_attribute_labels("channel", channels)
    list = os.listdir(input_dir + "/GFP/")
    number_files = len(list)

    for t in range(1, number_files + 1):
        for j in range(len(channels)):
            stack = imread(
                input_dir + channels[j] + "/t" + f"{t:04}" + "_" + channels[j] + ".tif"
            ).astype(np.uint16)
            bdv_writer.append_view(
                stack,
                time=t,
                channel=j,
                voxel_size_xyz=(0.347, 0.347, 2),
                voxel_units="um",
            )
    bdv_writer.write_xml()
    bdv_writer.close()
    print(f"dataset in {fname}")


if __name__ == "__main__":
    app.run(main)
