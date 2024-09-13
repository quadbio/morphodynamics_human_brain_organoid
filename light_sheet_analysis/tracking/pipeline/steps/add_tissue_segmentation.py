import numpy as np
import skimage
import zarr
from absl import app, flags
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels
from skimage.io import imread
from skimage.transform import resize

FLAGS = flags.FLAGS

flags.DEFINE_string("mask_dir", "", "Directory of the tissue segmentation")
flags.DEFINE_string("input_path", "", "Input path of the omezarr file")


def add_tissue_masks(
    time_point,
    label,
    mask_dir,
    channel_group,
    shape_input,
    chunk_size,
):
    combined_masks = imread(
        mask_dir + f"{(time_point+1):04}" + "_lumen_organoid_mask_processed.tif"
    )

    time_group = channel_group.require_group(str(time_point))
    write_labels(
        labels=resize(combined_masks, shape_input, anti_aliasing=False, order=0),
        group=time_group,
        name="tissue_mask",
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )
    organoid_mask = (combined_masks >= 2).astype(int)

    lumen_mask = skimage.morphology.remove_small_objects(
        (combined_masks == 3), min_size=20000 / (0.347 * 0.347 * 2 * 4 * 4)
    )

    # Number of Lumen
    label_img, num_labels = skimage.measure.label(lumen_mask, return_num=True)
    resized = resize(label_img, shape_input, anti_aliasing=False, order=0)
    write_labels(
        labels=resized,
        group=time_group,
        name=label,
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )


def main(argv):
    mask_dir = FLAGS.mask_dir
    zarr_path = FLAGS.input_path
    zarr_array = zarr.open(zarr_path, mode="r")
    zarr_array["GFP"]
    label = "lumen_masks"
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store)
    channels = list(zarr_array.group_keys())
    for channel_name in channels:
        input_movie = zarr_array[channel_name]
        input_shape = input_movie["0"]["0"].shape
        size_z = input_shape[0]
        size_x = input_shape[1]
        size_y = input_shape[2]
        time_points = np.arange(125)

        channel_group = root.require_group(channel_name)
        chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)
        result = Parallel(n_jobs=8, backend="multiprocessing", verbose=5)(
            delayed(add_tissue_masks)(
                time_point,
                label,
                mask_dir,
                channel_group,
                shape_input=input_shape,
                chunk_size=chunk_size,
            )
            for time_point in time_points
        )


if __name__ == "__main__":
    app.run(main)
