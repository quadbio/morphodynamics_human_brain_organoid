import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from morphometrics.measure import measure_selected
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rescale
from tqdm import tqdm


def measure_one_mask(im, label, im_mask, measurement_selection):
    im_mask = np.pad(im_mask, 16)
    im = np.pad(im, 16)
    one_measurement = measure_selected(
        label_image=im_mask,
        intensity_image=im,
        measurement_selection=measurement_selection,
        verbose=False,
    )
    return one_measurement


def extract_morphometrics(
    channel,
    time_point,
    channel_name,
    output_dir,
    label_name="cell_segmentation",
    n_jobs=32,
    zarr_level="0",
):
    cell_mask = channel[str(time_point)]["labels"][label_name][zarr_level][:]
    cell_image = channel[str(time_point)][zarr_level][:]
    zarr_level = int(zarr_level)
    if len(np.unique(cell_mask)) > 1:
        # run multithreaded morphology analysis
        measurement_selection = [
            "surface_properties_from_labels",
            {
                "name": "regionprops",
                "choices": {
                    "size": True,
                    "intensity": True,
                    "position": True,
                    "moments": True,
                },
            },
        ]
        cell_image = rescale(
            cell_image,
            [(2 / (0.347 * 2 ** (zarr_level + 1))), 1, 1],
            anti_aliasing=False,
            preserve_range=True,
        )
        cell_mask = rescale(
            cell_mask,
            [(2 / (0.347 * 2 ** (zarr_level + 1))), 1, 1],
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.uint16)

        region_properties_table = regionprops_table(
            cell_mask,
            intensity_image=cell_image,
            properties=(
                "label",
                "bbox",
                "intensity_image",
                "centroid",
                "moments",
                "moments_normalized",
                "moments_central",
                "weighted_centroid",
            ),
        )
        region_properties_mask = pd.DataFrame(
            regionprops_table(
                cell_mask,
                intensity_image=cell_mask,
                properties=("label", "bbox", "intensity_image"),
            )
        )
        major_axis = []
        minor_axis = []

        for region in tqdm(regionprops(cell_mask)):
            major_axis.append(region.axis_major_length)
            try:
                minor_axis.append(region.axis_minor_length)
            except:
                minor_axis.append(np.nan)
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=1)(
            delayed(measure_one_mask)(im, label, im_mask, measurement_selection)
            for im, label, im_mask in zip(
                region_properties_table["intensity_image"],
                region_properties_table["label"],
                region_properties_mask["intensity_image"],
            )
        )

        all_measurements = pd.DataFrame()
        for result in results:
            all_measurements = pd.concat([all_measurements, result])
        centroid_cols = [col for col in all_measurements.columns if "centroid" in col]
        bbox_cols = [col for col in all_measurements.columns if "bbox-" in col]
        moments_cols = [col for col in all_measurements.columns if "moments" in col]
        all_redo_cols = centroid_cols + bbox_cols + moments_cols
        for col in all_redo_cols:
            all_measurements[col] = region_properties_table[col]
        all_measurements["axis_minor_length"] = minor_axis
        all_measurements["axis_major_length"] = major_axis
        # all_measurements['label']=all_measurements.index
        all_measurements = all_measurements[all_measurements["area"] > 100]
        all_measurements.to_csv(output_dir + f"{channel_name}_{time_point}.csv")
