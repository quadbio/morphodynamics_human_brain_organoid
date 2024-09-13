import numpy as np
import zarr
from absl import app, flags
from joblib import Parallel, delayed
from lightinsight.registration.registration import (
    calculate_cummulative_translations,
    register_frame,
    register_labels,
)
from ome_zarr.io import parse_url

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", "", "Input path of the omezarr file")
flags.DEFINE_string("channel", "", "Channel to use as the leading channel")


def main(argv):
    zarr_path = FLAGS.input_path
    leading_channel_name = FLAGS.channel
    zarr_array = zarr.open(zarr_path, mode="r")
    leading_channel = zarr_array[leading_channel_name]
    time_points = sorted(np.array(list(leading_channel.group_keys())).astype(int))

    cum_sum_translations, padding = calculate_cummulative_translations(
        leading_channel,
        pyramid_order="0",
        method="phase_cross_correlation",
        upsample_factor=20,
        num_comparisons=3,
        percentile=None,
        time_points=time_points,
        n_jobs=8,
    )
    out_path = zarr_path.replace(".zarr", "_registered.zarr")
    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)
    channels = list(zarr_array.group_keys())
    for channel_name in channels:
        input_movie = zarr_array[channel_name]

        input_shape = input_movie["0"]["0"].shape + np.array(padding).sum(1)
        size_z = input_shape[0]
        size_x = input_shape[1]
        size_y = input_shape[2]
        time_points = sorted(np.array(list(input_movie.group_keys())).astype(int))

        channel_group = root.require_group(channel_name)
        chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)
        result = Parallel(n_jobs=8, backend="multiprocessing", verbose=5)(
            delayed(register_frame)(
                time_point,
                input_movie,
                channel_group,
                padding,
                chunk_size,
                cum_sum_translations,
            )
            for time_point in time_points
        )

        # Create list of time points with labels

        time_points_labels = [
            t for t in time_points if "labels" in list(input_movie[str(t)].group_keys())
        ]

        if len(time_points_labels) > 0:
            labels = list(input_movie["0"]["labels"].group_keys())

            for label in labels:
                # Create list of time points for each label
                time_points_label = [
                    t
                    for t in time_points_labels
                    if label in list(input_movie[str(t)]["labels"].group_keys())
                ]
                # Register time points
                result = Parallel(n_jobs=8, backend="multiprocessing", verbose=5)(
                    delayed(register_labels)(
                        time_point,
                        label,
                        input_movie,
                        channel_group,
                        padding,
                        chunk_size,
                        cum_sum_translations,
                    )
                    for time_point in time_points_label
                )


if __name__ == "__main__":
    app.run(main)
