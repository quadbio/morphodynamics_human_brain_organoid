import json
import os

import dask.array as da
import numpy as np

# Import libraries
import zarr
from absl import app, flags
from dask.distributed import Client, progress
from EmbedSeg.utils.create_dicts import create_test_configs_dict
from joblib import Parallel, delayed
from lightinsight.segmentation.embedseg import segment_image
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

# from dexp.datasets import ZDataset
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", "", "Input path of the omezarr file")
flags.DEFINE_string("array_section", "", "Section to segment")


def main(argv):
    zarr_path = FLAGS.input_path
    array_section = FLAGS.array_section

    tta = True
    ap_val = 0.5
    save_dir = "/cluster/project/treutlein/DATA/imaging/viventis/test_Embedseg/"
    data_dir = "/cluster/project/treutlein/DATA/imaging/viventis/test_Embedseg/"
    base_dir = "/cluster/home/gutgi/git_repositories/morphodynamics-of-human-brain-organoid-patterning/light_sheet_analysis/light_sheet_image_pipeline/cell_segmentation"
    project_name = "3D_Brain_organoids_with_meta"
    run_name = "all_06_02_2023"
    checkpoint_path = os.path.join(
        f"{base_dir}/experiment", project_name + "-" + run_name, "best_iou_model.pth"
    )

    if os.path.isfile(f"{base_dir}/data_properties_{run_name}.json"):
        with open(
            os.path.join(f"{base_dir}/data_properties_{run_name}.json")
        ) as json_file:
            data = json.load(json_file)
            data["one_hot"]
            data_type = data["data_type"]
            min_object_size = int(data["min_object_size"])
            # foreground_weight = float(data['foreground_weight'])
            # n_z, n_y, n_x = int(data['n_z']),int(data['n_y']), int(data['n_x'])
            pixel_size_z_microns, pixel_size_y_microns, pixel_size_x_microns = (
                float(data["pixel_size_z_microns"]),
                float(data["pixel_size_y_microns"]),
                float(data["pixel_size_x_microns"]),
            )
            # mask_start_x, mask_start_y, mask_start_z = 700,700,160
            # mask_end_x, mask_end_y, mask_end_z =  800,800,200
    if os.path.isfile(f"{base_dir}/normalization_{run_name}.json"):
        with open(
            os.path.join(f"{base_dir}/normalization_{run_name}.json")
        ) as json_file:
            data = json.load(json_file)
            norm = data["norm"]
    n_x = 600
    n_y = 600
    n_z = 80

    test_configs = create_test_configs_dict(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        tta=tta,
        ap_val=ap_val,
        min_object_size=min_object_size,
        save_dir=save_dir,
        norm=norm,
        data_type=data_type,
        n_z=n_z,
        n_y=n_y,
        type="test_Embedseg",
        n_x=n_x,
        anisotropy_factor=pixel_size_z_microns / pixel_size_x_microns,
        name="3d",
        seed_thresh=0.7,
        fg_thresh=0.4,
        expand_grid=False,
    )

    # Define input zarr path
    zarr_array = zarr.open(zarr_path, mode="r")
    channels = list(zarr_array.group_keys())

    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store)
    for channel in channels:
        time_points_all = list(zarr_array[channel].group_keys())
        # Split into 6, as we are using 6 gpus at the same time
        time_points = list(np.array_split(time_points_all, 12)[int(array_section)])
        print(time_points)
        size_z, size_x, size_y = zarr_array[channel]["0"]["0"].shape
        # channel_name=f"{channel}_cell_segmentation"
        channel_group = root.require_group(channel)
        chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)
        result = Parallel(n_jobs=3, backend="multiprocessing", verbose=5)(
            delayed(segment_image)(
                time_point, test_configs, channel_group, zarr_array, channel, chunk_size
            )
            for time_point in time_points
        )


if __name__ == "__main__":
    app.run(main)
