import numpy as np
import scipy
import skimage
import zarr
from absl import app, flags
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels
from tqdm import tqdm

FLAGS = flags.FLAGS


def write_empty_labels(time_point, label_name, channel_group, chunk_size, shape, dtype):

    # Create time group
    time_group = channel_group.require_group(str(time_point))
    # write small lumen masks to zarr
    write_labels(
        labels=np.zeros(shape, dtype=dtype),
        group=time_group,
        name=label_name,
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )


def extract_filtered(
    z,
    label,
    t_kernel_length,
    label_name_smooth,
    n_split,
    zarr_level,
    input_movie,
    time_points_label,
):
    z_split = np.array_split(
        np.arange(input_movie[0]["labels"][label][zarr_level].shape[0]), n_split
    )[z]
    zarr_shape = input_movie[0]["labels"][label][zarr_level][z_split, :, :].shape
    updated_movie = np.zeros((len(time_points_label),) + zarr_shape, dtype=np.uint8)
    for t in time_points_label:
        updated_movie[t, :, :, :] = input_movie[t]["labels"][label][zarr_level][
            z_split, :, :
        ]
    updated_movie = updated_movie == 3
    return scipy.ndimage.median_filter(updated_movie, size=(t_kernel_length, 1, 1, 1))

    # for t in time_points_label:
    #    input_movie[t]['labels'][label_name_smooth][zarr_level][z_split,:,:]=filtered[t]


def write_updated_labels(
    time_point,
    label_name,
    channel_group,
    chunk_size,
    zarr_level,
    input_movie,
    label_name_smooth
    # movie_filtered
):

    time_group = channel_group.require_group(str(time_point))
    # Load lumen mask + remove small objects

    lumen_mask = input_movie[time_point]["labels"][label_name_smooth][zarr_level][
        :
    ].astype(bool)
    # lumen_mask=movie_filtered[time_point]
    lumen_mask = scipy.ndimage.binary_opening(lumen_mask, iterations=4)
    lumen_mask = skimage.morphology.remove_small_objects(
        lumen_mask,
        min_size=20000
        / (0.347 * 0.347 * 2 * (2 ** (zarr_level + 1)) * (2 ** (zarr_level + 1))),
    )
    lumen_mask = skimage.measure.label(lumen_mask)

    # write small lumen masks to zarr
    write_labels(
        labels=lumen_mask,
        group=time_group,
        name=label_name,
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )


flags.DEFINE_string("input_path", "", "Input path of the omezarr file")
flags.DEFINE_string("label", "", "Which label to segment")
flags.DEFINE_string("channel", "", "Which channel to segment")
flags.DEFINE_string("kernel_size", "", "Length of the temporal filter")


def main(argv):
    zarr_path = FLAGS.input_path
    channel = FLAGS.channel
    label = FLAGS.label
    smootheness = int(FLAGS.kernel_size)
    zarr_level = 0
    n_split = 20
    t_kernel_length = smootheness

    label_name_smooth = f"lumen_masks_temp_smooth_{smootheness}"
    zarr_array = zarr.open(zarr_path, mode="r+")
    channels = list(zarr_array.group_keys())
    list(zarr_array[channel]["0"]["labels"].group_keys())
    input_movie = zarr_array[channel]
    time_points = sorted(np.array(list(input_movie.group_keys())).astype(int))
    time_points_labels = [
        t for t in time_points if "labels" in list(input_movie[str(t)].group_keys())
    ]
    time_points_label = [
        t
        for t in time_points_labels
        if label in list(input_movie[str(t)]["labels"].group_keys())
    ]

    if not label_name_smooth in list(input_movie[0]["labels"].group_keys()):
        # print: adding empty array
        store = parse_url(zarr_path, mode="w").store
        root = zarr.group(store=store)
        channels = list(zarr_array.group_keys())
        dtype = np.uint8
        for channel_name in channels:
            channel_group = root.require_group(channel_name)
            input_movie = zarr_array[channel_name]

            # calculate chunk size
            input_shape = input_movie["0"]["0"].shape
            size_z = input_shape[0]
            size_x = input_shape[1]
            size_y = input_shape[2]
            chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)

            # write smooth time points time points
            result = Parallel(n_jobs=8, backend="multiprocessing", verbose=2)(
                delayed(write_empty_labels)(
                    time_point=time_point,
                    label_name=label_name_smooth,
                    channel_group=channel_group,
                    chunk_size=chunk_size,
                    shape=input_shape,
                    dtype=dtype,
                )
                for time_point in time_points_label
            )

    movie_filtered = Parallel(n_jobs=8, verbose=8)(
        delayed(extract_filtered)(
            z,
            label,
            t_kernel_length,
            label_name_smooth,
            n_split,
            zarr_level,
            input_movie,
            time_points_label,
        )
        for z in range(n_split)
    )

    movie_filtered = np.concatenate(movie_filtered, axis=1)
    for t in tqdm(time_points_label):
        input_movie[t]["labels"][label_name_smooth][zarr_level][:] = movie_filtered[t]
    movie_filtered = []

    store = parse_url(zarr_path, mode="r+").store
    root = zarr.group(store=store)
    label_name = f"lumen_masks_smooth_{smootheness}_processed"
    channels = list(zarr_array.group_keys())

    for channel_name in channels:

        channel_group = root.require_group(channel_name)

        input_movie = zarr_array[channel_name]

        # calculate chunk size
        input_shape = input_movie["0"]["0"].shape
        size_z = input_shape[0]
        size_x = input_shape[1]
        size_y = input_shape[2]
        chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)

        # write smooth time points time points
        result = Parallel(n_jobs=8, backend="multiprocessing", verbose=6)(
            delayed(write_updated_labels)(
                time_point,
                label_name,
                channel_group,
                chunk_size,
                zarr_level,
                input_movie,
                label_name_smooth
                # movie_filtered
            )
            for time_point in time_points_label
        )


if __name__ == "__main__":
    app.run(main)
