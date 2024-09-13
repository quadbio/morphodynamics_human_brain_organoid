import numpy as np
import zarr
from ome_zarr.writer import write_labels
from skimage import morphology


def create_detections(
    time_point, input_movie, channel_group, cumsum_max, label, chunk_size
):
    mask = input_movie[str(time_point)]["labels"][label]["0"][:].astype(np.uint32)
    detections = mask + cumsum_max[time_point]
    detections = detections * (mask > 0)
    detections = morphology.remove_small_objects(detections, 1500)
    time_group = channel_group.require_group(str(time_point))
    write_labels(
        labels=detections,
        group=time_group,
        name=label + "_detection",
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )
