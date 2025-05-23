import joblib
import numpy as np
import pandas as pd
import pymeshfix
import scipy.spatial
from joblib import Parallel, delayed
from lightinsight.angle.cell_angle import ellipsoid_fit
from morphometrics.measure import measure_selected
from skimage.measure import label, marching_cubes, regionprops, regionprops_table
from skimage.transform import rescale
from sklearn import metrics
from tqdm import tqdm
from trimesh import Trimesh
from trimesh.smoothing import filter_taubin


def extract_mesh(mask):
    vertices, faces, _, _ = marching_cubes(mask, 0, step_size=40)
    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)
    organoid_mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)
    filter_taubin(organoid_mesh, iterations=50)

    surface_normal_starts = organoid_mesh.vertices
    surface_normals = organoid_mesh.vertex_normals
    return organoid_mesh, surface_normal_starts, surface_normals


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


def extract_angles(
    organoid_mask,
    zarr_level,
    cell_properties_df=None,
    cell_image_df=None,
    cell_mask=None,
    cell_image=None,
    rescale_cell=True,
    use_cell_image=True,
):
    """
    Extract angles between cell major axis and surface normals

    Parameters:
    -----------
    organoid_mask : numpy.ndarray
        Mask of the organoid (values > 1 are considered part of the organoid)
    cell_properties_df : pandas.DataFrame
        DataFrame with cell properties including centroid and intensity_image
    cell_image_df : pandas.DataFrame
        DataFrame with cell properties: intensity_image

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added columns for angle, max_radii, min_radii, etc.
    """

    # Preprocess organoid mask
    organoid_mask = organoid_mask > 1
    organoid_mask = rescale(
        organoid_mask,
        [(2 / (0.347 * 2 ** (zarr_level + 1))), 1, 1],
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.uint16)

    # Extract organoid mesh and surface normals
    organoid_mesh, surface_normal_starts, surface_normals = extract_mesh(organoid_mask)

    # Initialize lists to store results
    all_angles = []
    max_radii = []
    min_radii = []
    radii_all = []
    evecs_all = []
    evals_all = []

    if cell_properties_df is None:
        cell_properties_df = pd.DataFrame(
            regionprops_table(
                cell_mask,
                intensity_image=cell_image,
                properties=("label", "intensity_image", "image", "centroid"),
            )
        )

    counter = 0
    for label in cell_properties_df.index.unique():
        counter += 1
        try:
            cell_row = cell_properties_df[cell_properties_df.index == label].iloc[0]
            if cell_image_df is not None and not cell_image_df.empty:
                cell_row_im = cell_image_df[cell_image_df["label"] == label].iloc[0]
                if use_cell_image:
                    mask_1 = cell_row_im["intensity_image"].astype(int) > 0
                else:
                    mask_1 = cell_row_im["image"].astype(int) > 0
            else:
                if use_cell_image:
                    mask_1 = cell_row["intensity_image"].astype(int) > 0
                else:
                    mask_1 = cell_row["image"].astype(int) > 0

            if rescale_cell:
                mask_1 = rescale(
                    mask_1,
                    [(2 / (0.347 * 2 ** (zarr_level + 1))), 1, 1],
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                ).astype(np.uint16)

            # Extract cell mesh and fit ellipsoid
            vertices, faces, _, _ = marching_cubes(mask_1, 0, step_size=2)
            vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)
            center, evecs, radii, v, evals = ellipsoid_fit(vertices_clean)

            # Use the cell centroid from properties
            center = [
                cell_row["centroid-0"],
                cell_row["centroid-1"],
                cell_row["centroid-2"],
            ]

            # Find closest surface normal
            KD_tree_surface_normals = scipy.spatial.cKDTree(
                surface_normal_starts, leafsize=100
            )
            nearest_neighbor = KD_tree_surface_normals.query(center, k=1)

            # Calculate angle between primary cell axis and surface normal
            primary_eigenvector = evecs[np.where(radii == max(radii))[0][0]]
            cosine_sim = abs(
                metrics.pairwise.cosine_similarity(
                    [primary_eigenvector], [surface_normals[nearest_neighbor[1]]]
                )[0][0]
            )
            all_angles.append(cosine_sim)
            max_radii.append(max(radii))
            min_radii.append(min(radii))
            radii_all.append(radii)
            evecs_all.append(evecs)
            evals_all.append(evals)

        except Exception as e:
            print(f"Couldn't fit ellipsoid: {e}")
            max_radii.append(np.nan)
            min_radii.append(np.nan)
            radii_all.append(np.nan)
            evecs_all.append(np.nan)
            evals_all.append(np.nan)
            all_angles.append(np.nan)

    cell_properties_df["angle"] = all_angles
    cell_properties_df["max_radii"] = max_radii
    cell_properties_df["min_radii"] = min_radii
    cell_properties_df["radii"] = radii_all
    cell_properties_df["evecs"] = evecs_all
    cell_properties_df["evals"] = evals_all

    return cell_properties_df


def extract_morphometrics(
    channel,
    time_point,
    channel_name,
    output_dir,
    label_name="cell_segmentation",
    n_jobs=32,
    zarr_level="0",
    rf_save_dir=None,
    feature_list_path=None,
    tissue_label=None,
    calculate_angles=False,
    upscale_before_angles=False,
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
        cell_image_rescaled = rescale(
            cell_image,
            [(2 / (0.347 * 2 ** (zarr_level + 1))), 1, 1],
            anti_aliasing=False,
            preserve_range=True,
        )
        cell_mask_rescaled = rescale(
            cell_mask,
            [(2 / (0.347 * 2 ** (zarr_level + 1))), 1, 1],
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.uint16)

        region_properties_table = regionprops_table(
            cell_mask_rescaled,
            intensity_image=cell_image_rescaled,
            properties=(
                "label",
                "bbox",
                "image",
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
                cell_mask_rescaled,
                intensity_image=cell_mask_rescaled,
                properties=("label", "bbox", "intensity_image"),
            )
        )
        major_axis = []
        minor_axis = []
        labels = []

        for region in tqdm(regionprops(cell_mask_rescaled)):
            labels.append(region.label)
            major_axis.append(region.axis_major_length)
            try:
                minor_axis.append(region.axis_minor_length)
            except Exception as exc:
                minor_axis.append(np.nan)
                print(f"Couldn't fin minor axis: {exc}")

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
        all_redo_cols = (
            centroid_cols + bbox_cols + moments_cols + ["intensity_image", "image"]
        )
        for col in all_redo_cols:
            all_measurements[col] = region_properties_table[col]
        all_measurements = all_measurements.copy()

        assert (labels == all_measurements.index).all()
        all_measurements["axis_minor_length"] = minor_axis
        all_measurements["axis_major_length"] = major_axis
        all_measurements = all_measurements[all_measurements["area"] > 100]

        if rf_save_dir is not None and feature_list_path is not None:
            # Load the random forest classifier
            grid_clf = joblib.load(rf_save_dir)
            feature_list = pd.read_csv(feature_list_path)

            # Get the features from the 'features' column
            expected_features = feature_list["features"].tolist()
            all_measurements["channel"] = channel_name
            all_measurements["channel"] = (all_measurements["channel"] == "GFP").astype(
                int
            )
            # Check which features are available in the current data
            available_features = [
                f for f in expected_features if f in all_measurements.columns
            ]
            missing_features = [
                f for f in expected_features if f not in all_measurements.columns
            ]

            if missing_features:
                print(f"Warning: Missing features: {missing_features}")

            if available_features:
                # Extract feature matrix in the exact order expected by the model
                X_all = all_measurements[expected_features].copy()
                X_all = X_all.dropna(axis=0, how="any")
                X_all = np.array(X_all)

                # Predict cell types
                predicted_labels = grid_clf.predict(X_all)
                all_measurements["structure_labels"] = predicted_labels

                # Store prediction probabilities
                probabilities = grid_clf.predict_proba(X_all)
                all_measurements["prediction_confidence"] = np.max(
                    probabilities, axis=1
                )

            else:
                raise ValueError(
                    f"None of the expected features found in the data. Expected: {expected_features}"
                )

        if calculate_angles and tissue_label is not None:
            organoid_mask = channel[str(time_point)]["labels"][tissue_label][
                zarr_level
            ][:]

            if upscale_before_angles:
                all_measurements = extract_angles(
                    organoid_mask=organoid_mask,
                    zarr_level=zarr_level,
                    cell_properties_df=all_measurements,
                    rescale_cell=False,
                )

            else:
                original_cell_properties = pd.DataFrame(
                    regionprops_table(
                        cell_mask,
                        intensity_image=cell_image,
                        properties=("label", "centroid", "image", "intensity_image"),
                    )
                )

                all_measurements = extract_angles(
                    organoid_mask=organoid_mask,
                    zarr_level=zarr_level,
                    cell_image_df=original_cell_properties,
                    cell_properties_df=all_measurements,
                )

        all_measurements.to_csv(output_dir + f"{channel_name}_{time_point}.csv")
