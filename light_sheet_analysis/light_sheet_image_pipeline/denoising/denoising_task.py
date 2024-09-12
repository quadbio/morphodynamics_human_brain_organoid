import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
import os

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from absl import app, flags
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.utils import plot_history
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2V, N2VConfig
from n2v.utils.n2v_utils import manipulate_val_data
from skimage.io import imread, imsave

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_folder",
    "",
    "Input folder with the TIFF stacks",
)
flags.DEFINE_string(
    "output_folder",
    "",
    "Output folder",
)
flags.DEFINE_string(
    "config_file",
    "/cluster/home/gutgi/light_sheet_image_pipeline/pipeline_config.yaml",
    "configuration file to use",
)
flags.DEFINE_string("model_dir", "", "directory for the denoising model")


flags.DEFINE_string("f", "", "kernel")

flags.DEFINE_string("channel", "", "Channel to denoise")

flags.DEFINE_string("jobname", "test", "jobname")


def main(argv):
    # flags used
    config_file = FLAGS.config_file
    input_dir = FLAGS.input_folder
    output_dir = FLAGS.output_folder
    channel = FLAGS.channel
    denoising_model_basedir = FLAGS.model_dir

    # Import config
    name_folder = input_dir.split("/")[-2]
    import yaml

    with open(config_file, "r") as stream:
        configuration = yaml.safe_load(stream)

    n_random_images = configuration["denoising"]["n_random_images"]
    n_epochs = configuration["denoising"]["n_epochs"]
    test_train_split = configuration["denoising"]["test_train_split"]
    denoise_clip = configuration["denoising"]["denoise_clip"]
    model_name = "n2v_2D_" + channel + name_folder + "denoised_movie"

    if not os.path.exists(output_dir + channel + "/"):
        os.makedirs(output_dir + channel + "/")

    # Get random slices from whole movie
    t = 1
    image_dims = imread(f"{input_dir}{channel}/t{t:04}_{channel}.tif").shape
    z_dim = image_dims[0]
    list = os.listdir(input_dir + "/" + channel + "/")
    n_files = len(list)
    n_slices = z_dim * n_files
    random_slices_ind = np.random.randint(0, n_slices, size=n_random_images)
    random_slices = []
    for random_slice_ind in random_slices_ind:
        t = (random_slice_ind // z_dim) + 1
        z_slice = random_slice_ind % z_dim
        image = imread(f"{input_dir}{channel}/t{t:04}_{channel}.tif")
        random_slices.append(image[z_slice].copy())
        image = []

    random_slices = np.array(random_slices)

    # Train N2V
    datagen = N2V_DataGenerator()
    patch_shape = (96, 96)
    random_slices = random_slices.reshape((1,) + random_slices.shape + (1,))
    X = datagen.generate_patches_from_list(
        random_slices[:, : int(len(random_slices[0]) * test_train_split)],
        shape=patch_shape,
    )
    X_val = datagen.generate_patches_from_list(
        random_slices[
            :,
            -(len(random_slices[0]) - int(len(random_slices[0]) * test_train_split)) :,
        ],
        shape=patch_shape,
    )

    config = N2VConfig(
        X,
        unet_kern_size=3,
        train_steps_per_epoch=int(X.shape[0] / 256),
        train_epochs=n_epochs,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=256,
        n2v_perc_pix=0.198,
        n2v_patch_shape=(96, 96),
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
    )

    # We are now creating our network model.
    model = N2V(config=config, name=model_name, basedir=denoising_model_basedir)
    history = model.train(X, X_val)
    plt.figure(figsize=(16, 5))
    history_plot = plt.figure(figsize=(16, 5))

    plt.plot(history.history["loss"], ".-")
    plt.plot(history.history["val_loss"], ".-")
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    history_plot.figure.savefig(
        denoising_model_basedir
        + "/"
        + model_name
        + "/"
        + model_name
        + "_training_curve.png"
    )
    pd.DataFrame(history.history).to_csv(
        denoising_model_basedir
        + "/"
        + model_name
        + "/"
        + model_name
        + "_training_curve.csv"
    )

    model.export_TF(
        name="Noise2Void viventis 2D",
        description=f"This is the 2D Noise2Void trained on {n_random_images} images.",
        authors=["Gilles Gut"],
        test_img=X_val[0, ..., 0],
        axes="YX",
        patch_shape=patch_shape,
    )

    model = N2V(config=None, name=model_name, basedir=denoising_model_basedir)

    print("model trained")
    for t in range(1, n_files + 1):
        image = imread(f"{input_dir}{channel}/t{t:04}_{channel}.tif")
        image = image.astype(np.float32)
        denoised_image = []
        for i in range(len(image)):
            denoised_image.append(model.predict(image[i], axes="YX"))
        denoised_image = np.array(denoised_image)
        bk = denoised_image.min(axis=0)
        denoised_image_no_bk = denoised_image - (bk + denoise_clip)
        denoised_image_no_bk = denoised_image_no_bk.clip(0.0, 2**16 - 1)
        print(
            np.count_nonzero(denoised_image_no_bk == 0)
            / (image.shape[0] * image.shape[1] * image.shape[2])
        )
        denoised_image_no_bk = denoised_image_no_bk.astype(np.uint16)
        imsave(
            f"{output_dir}{channel}/t{t:04}_{channel}.tif",
            denoised_image_no_bk,
            plugin="tifffile",
            check_contrast=False,
            compress=6,
            bigtiff=True,
        )


if __name__ == "__main__":
    app.run(main)
