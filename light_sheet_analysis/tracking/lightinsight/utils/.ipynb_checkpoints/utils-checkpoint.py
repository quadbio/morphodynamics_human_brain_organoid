import numpy as np
import zarr


def pad_to_shape(a, shape):
    y_, x_, z_ = shape
    y, x, z = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    z_pad = z_ - z
    return np.pad(
        a,
        (
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (x_pad // 2, x_pad // 2 + x_pad % 2),
            (z_pad // 2, z_pad // 2 + z_pad % 2),
        ),
        mode="constant",
    )


def max_frame(time_point, input_movie, label):
    return input_movie[str(time_point)]["labels"][label]["0"][:].max()
