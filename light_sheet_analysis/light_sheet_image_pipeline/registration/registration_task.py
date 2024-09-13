import os

import itk
import numpy as np
from absl import app, flags
from registration import rescale_intensity, rigid_registration_image
from skimage.io import imread, imsave

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_folder",
    "/cluster/project/treutlein/DATA/imaging/viventis/test_pipeline/",
    "Input folder with the TIFF stacks",
)

flags.DEFINE_string(
    "output_folder",
    "/cluster/work/treutlein/DATA/imaging/viventis/test_pipeline/denoised/",
    "Output folder",
)

flags.DEFINE_string(
    "config_file",
    "/cluster/home/gutgi/light_sheet_image_pipeline/pipeline_config.yaml",
    "configuration file to use",
)

flags.DEFINE_string("f", "", "kernel")

flags.DEFINE_string("jobname", "test", "jobname")

flags.DEFINE_string("wait", "", "wait_argument")


def main(argv):
    # flags used
    FLAGS.config_file
    input_dir = FLAGS.input_folder
    output_dir = FLAGS.output_folder
    channels = ["GFP", "mCherry"]
    padding = ((32, 32), (96, 96), (96, 96))
    list = os.listdir(input_dir + "/GFP/")
    number_files = len(list)

    counter = 0
    for t in range(1, number_files + 1):
        if not os.path.exists(output_dir + channels[0] + "/"):
            os.makedirs(output_dir + channels[0] + "/")
        if not os.path.exists(output_dir + channels[1] + "/"):
            os.makedirs(output_dir + channels[1] + "/")

        # Transform Channel 1
        input_image = imread(
            input_dir + channels[0] + "/t" + f"{t:04}" + "_" + channels[0] + ".tif"
        ).astype(np.float32)
        input_image = np.pad(input_image, padding)
        # moving_image = itk.image_view_from_array(moving_image)
        moving_image = itk.image_view_from_array(rescale_intensity(input_image))

        if counter > 0:
            # new_image, transformation = rigid_registration_image(
            #    fixed_image, moving_image
            # )
            fixed_image, transformation = rigid_registration_image(
                fixed_image, moving_image
            )
            new_image = itk.transformix_filter(
                itk.image_view_from_array(input_image), transformation
            )

        else:
            new_image = input_image
        new_image = np.asarray(new_image).clip(0.0, 2**16 - 1).astype(np.uint16)
        # fixed_image = itk.image_view_from_array(new_image.astype(np.float32))
        # Save Channel 1
        imsave(
            output_dir
            + channels[0]
            + "/"
            + "t"
            + f"{t:04}"
            + "_"
            + channels[0]
            + ".tif",
            new_image,
            plugin="tifffile",
            check_contrast=False,
            compress=6,
            bigtiff=True,
        )

        print("loaded channel 1")
        new_image = []

        # Load, transform and save Channel 2
        moving_channel_2 = imread(
            input_dir + channels[1] + "/t" + f"{t:04}" + "_" + channels[1] + ".tif"
        ).astype(np.float32)
        moving_channel_2 = np.pad(moving_channel_2, padding)
        moving_channel_2 = itk.image_view_from_array(moving_channel_2)
        print("loaded channel 2")
        if counter > 0:
            new_image_2 = itk.transformix_filter(moving_channel_2, transformation)
        else:
            new_image_2 = moving_channel_2

        new_image_2 = np.asarray(new_image_2)
        new_image_2 = new_image_2.clip(0.0, 2**16 - 1)
        new_image_2 = new_image_2.astype(np.uint16)

        imsave(
            output_dir
            + channels[1]
            + "/"
            + "t"
            + f"{t:04}"
            + "_"
            + channels[1]
            + ".tif",
            new_image_2,
            plugin="tifffile",
            check_contrast=False,
            compress=6,
            bigtiff=True,
        )
        new_image_2 = []
        counter = counter + 1


if __name__ == "__main__":
    app.run(main)
