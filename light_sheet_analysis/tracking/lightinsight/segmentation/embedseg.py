import os

import torch
import torch.nn.functional as F
from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.generate_crops import (
    normalize_mean_std,
    normalize_min_max_percentile,
)
from EmbedSeg.utils.metrics import matching_dataset, obtain_APdsb_one_hot
from EmbedSeg.utils.utils import Cluster, Cluster_3d
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
import numpy as np
import ome_zarr
from EmbedSeg.utils.test_time_augmentation import apply_tta_2d, apply_tta_3d
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment, minimize_scalar
from skimage.segmentation import relabel_sequential
from tifffile import imsave


def segment_image(
    time_point, test_configs, channel_group, zarr_array, channel, chunk_size
):

    time_group = channel_group.require_group(time_point)

    image = zarr_array[channel][str(time_point)]["0"][:]
    image = image.astype(np.float32)
    instance_map, seed_map = begin_evaluating(test_configs, image)

    ome_zarr.writer.write_labels(
        labels=instance_map,
        group=time_group,
        name=f"cell_segmentation",
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )


"""
The code below is a modified version to directly segment 3D numpy arrays instead of 3D tiff files from the brilliant segmentation algorithm, EmbedSeg: https://github.com/juglab/EmbedSeg/tree/main!
"""


def convert_zyx_to_czyx(im):
    im = im[np.newaxis, ...]  # CZYX
    return im


def begin_evaluating(
    test_configs,
    image_input,
    optimize=False,
    maxiter=10,
    verbose=False,
    mask_region=None,
    device="cuda:0",
):
    """Entry function for inferring on test images

    Parameters
    ----------
    test_configs : dictionary
        Dictionary containing testing-specific parameters (for e.g. the `seed_thresh`  to use)
    optimize : bool, optional
        It is possible to determine the best performing `fg_thresh` by optimizing over different values on the validation sub-set
        By default and in the absence of optimization (i.e. `optimize=False`), the fg_thresh  is set equal to 0.5
    maxiter: int
        Number of iterations of optimization.
        Comes into play, only if `optimize=True`
    verbose: bool, optional
        If set equal to True, prints the AP_dsb for each image individually
    mask_region: list of lists, optional
        If a certain region of the image is not labelled in the GT label mask, that can be specified here.
        This enables comparison of the model prediction only with the area which is labeled in the GT label mask
    Returns
    -------
    result_dic: Dictionary
        Keys include the employed `fg_thresh` and the corresponding `AP_dsb` at IoU threshold = 0.5
    """
    n_sigma = test_configs["n_sigma"]
    ap_val = test_configs["ap_val"]
    min_mask_sum = test_configs["min_mask_sum"]
    min_unclustered_sum = test_configs["min_unclustered_sum"]
    min_object_size = test_configs["min_object_size"]
    test_configs["mean_object_size"]
    tta = test_configs["tta"]
    seed_thresh = test_configs["seed_thresh"]
    fg_thresh = test_configs["fg_thresh"]
    save_images = test_configs["save_images"]
    save_results = test_configs["save_results"]
    save_dir = test_configs["save_dir"]
    test_configs["anisotropy_factor"]
    grid_x = test_configs["grid_x"]
    grid_y = test_configs["grid_y"]
    grid_z = test_configs["grid_z"]
    pixel_x = test_configs["pixel_x"]
    pixel_y = test_configs["pixel_y"]
    pixel_z = test_configs["pixel_z"]
    one_hot = test_configs["dataset"]["kwargs"]["one_hot"]
    cluster_fast = test_configs["cluster_fast"]
    expand_grid = test_configs["expand_grid"]

    uniform_ds_factor = test_configs["dataset"]["kwargs"]["uniform_ds_factor"]
    normalization = test_configs["dataset"]["kwargs"]["normalization"]
    norm = test_configs["dataset"]["kwargs"]["norm"]
    data_type = test_configs["dataset"]["kwargs"]["data_type"]
    test_configs["dataset"]["kwargs"]["transform"]
    image = image_input  # ZYX
    if normalization and norm == "min-max-percentile":
        image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1, 2))
    elif normalization and norm == "mean-std":
        image = normalize_mean_std(image)
    elif normalization and norm == "absolute":
        image = image.astype(np.float32)
        if data_type == "8-bit":
            image /= 255
        elif data_type == "16-bit":
            image /= 65535
    image = convert_zyx_to_czyx(image)  # CZYX
    image = np.array([image])

    # transform
    image = torch.from_numpy(image.astype(np.float32))

    # set device
    device = torch.device(device if test_configs["cuda"] else "cpu")

    # load model
    model = get_model(test_configs["model"]["name"], test_configs["model"]["kwargs"])
    model = torch.nn.DataParallel(model).to(device)

    # load snapshot
    if os.path.exists(test_configs["checkpoint_path"]):
        state = torch.load(test_configs["checkpoint_path"])
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        assert False, "checkpoint_path {} does not exist!".format(
            test_configs["checkpoint_path"]
        )

    # test on evaluation images:
    result_dic = {}
    args = (
        seed_thresh,
        ap_val,
        min_mask_sum,
        min_unclustered_sum,
        min_object_size,
        tta,
        model,
        save_images,
        save_results,
        save_dir,
        verbose,
        grid_x,
        grid_y,
        grid_z,
        pixel_x,
        pixel_y,
        pixel_z,
        one_hot,
        mask_region,
        n_sigma,
        cluster_fast,
        expand_grid,
        uniform_ds_factor,
    )

    instance_map, seed_map = test_3d(fg_thresh, image, *args)
    result_dic["fg_thresh"] = fg_thresh
    result = 0
    result_dic["AP_dsb_05"] = -result
    return instance_map, seed_map


def stitch_3d(
    instance_map_tile,
    instance_map_current,
    z_tile=None,
    y_tile=None,
    x_tile=None,
    last=1,
    num_overlap_pixels=4,
):
    """Stitching instance segmentations together in case the full 3D image doesn't fit in one go, on the GPU
    This function is executed only if `expand_grid` is set to False

             Parameters
             ----------
             instance_map_tile : numpy array
                 instance segmentation over a tiled view of the image

             instance_map_current: numpy array
                 instance segmentation over the complete, large image

             z_tile: int
                 z position of the top left corner of the tile wrt the complete image

             y_tile: int
                 y position of the top left corner of the tile wrt the complete image

             x_tile: int
                 x position of the top left corner of the tile wrt the complete image

             last: int
                 number of objects currently present in the `instance_map_current`

             num_overlap_pixels: int
                 number of overlapping pixels while considering the next tile

             Returns
             -------
             tuple (int, numpy array)
                 (updated number of objects currently present in the `instance_map_current`,
                 updated instance segmentation over the full image)

    """

    mask = instance_map_tile > 0

    D = instance_map_tile.shape[0]
    H = instance_map_tile.shape[1]
    W = instance_map_tile.shape[2]

    instance_map_tile_sequential = np.zeros_like(instance_map_tile)

    if mask.sum() > 0:  # i.e. there were some object predictions
        # make sure that instance_map_tile is labeled sequentially

        ids, _, _ = relabel_sequential(instance_map_tile[mask])
        instance_map_tile_sequential[mask] = ids
        instance_map_tile = instance_map_tile_sequential

        # next pad the tile so that it is aligned wrt the complete image

        instance_map_tile = np.pad(
            instance_map_tile,
            (
                (z_tile, np.maximum(0, instance_map_current.shape[0] - z_tile - D)),
                (y_tile, np.maximum(0, instance_map_current.shape[1] - y_tile - H)),
                (x_tile, np.maximum(0, instance_map_current.shape[2] - x_tile - W)),
            ),
        )

        # ensure that it has the same shape as instance_map_current
        instance_map_tile = instance_map_tile[
            : instance_map_current.shape[0],
            : instance_map_current.shape[1],
            : instance_map_current.shape[2],
        ]
        mask_overlap = np.zeros_like(instance_map_tile)

        if z_tile == 0 and y_tile == 0 and x_tile == 0:
            ids_tile = np.unique(instance_map_tile)
            ids_tile = ids_tile[ids_tile != 0]
            instance_map_current[
                : instance_map_tile.shape[0],
                : instance_map_tile.shape[1],
                : instance_map_tile.shape[2],
            ] = instance_map_tile
            last = len(ids_tile) + 1
        else:
            if x_tile != 0 and y_tile == 0 and z_tile == 0:
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + H,
                    x_tile : x_tile + num_overlap_pixels,
                ] = 1
            elif x_tile == 0 and y_tile != 0 and z_tile == 0:
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + num_overlap_pixels,
                    x_tile : x_tile + W,
                ] = 1
            elif x_tile != 0 and y_tile != 0 and z_tile == 0:
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + H,
                    x_tile : x_tile + num_overlap_pixels,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + num_overlap_pixels,
                    x_tile : x_tile + W,
                ] = 1
            elif x_tile == 0 and y_tile == 0 and z_tile != 0:
                mask_overlap[
                    z_tile : z_tile + num_overlap_pixels,
                    y_tile : y_tile + H,
                    x_tile : x_tile + W,
                ] = 1
            elif x_tile != 0 and y_tile == 0 and z_tile != 0:
                mask_overlap[
                    z_tile : z_tile + num_overlap_pixels,
                    y_tile : y_tile + H,
                    x_tile : x_tile + W,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + H,
                    x_tile : x_tile + num_overlap_pixels,
                ] = 1
            elif x_tile == 0 and y_tile != 0 and z_tile != 0:
                mask_overlap[
                    z_tile : z_tile + num_overlap_pixels,
                    y_tile : y_tile + H,
                    x_tile : x_tile + W,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + num_overlap_pixels,
                    x_tile : x_tile + W,
                ] = 1
            elif x_tile != 0 and y_tile != 0 and z_tile != 0:
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + H,
                    x_tile : x_tile + num_overlap_pixels,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + num_overlap_pixels,
                    x_tile : x_tile + W,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + num_overlap_pixels,
                    y_tile : y_tile + H,
                    x_tile : x_tile + W,
                ] = 1

            # identify ids in the complete tile, not just the overlap region,
            ids_tile_all = np.unique(instance_map_tile)
            ids_tile_all = ids_tile_all[ids_tile_all != 0]

            # identify ids in the the overlap region,
            ids_tile_overlap = np.unique(instance_map_tile * mask_overlap)
            ids_tile_overlap = ids_tile_overlap[ids_tile_overlap != 0]

            # identify ids not in overlap region
            ids_tile_notin_overlap = np.setdiff1d(ids_tile_all, ids_tile_overlap)

            # identify ids in `instance_map_current` but only in the overlap region
            instance_map_current_masked = torch.from_numpy(
                instance_map_current * mask_overlap
            ).cuda()

            ids_current = (
                torch.unique(instance_map_current_masked).cpu().detach().numpy()
            )
            ids_current = ids_current[ids_current != 0]

            IoU_table = np.zeros((len(ids_tile_overlap), len(ids_current)))
            instance_map_tile_masked = torch.from_numpy(
                instance_map_tile * mask_overlap
            ).cuda()

            # rows are ids in tile, cols are ids in GT instance map

            for i, id_tile in enumerate(ids_tile_overlap):
                for j, id_current in enumerate(ids_current):

                    intersection = (
                        (instance_map_tile_masked == id_tile)
                        & (instance_map_current_masked == id_current)
                    ).sum()
                    union = (
                        (instance_map_tile_masked == id_tile)
                        | (instance_map_current_masked == id_current)
                    ).sum()
                    if union != 0:
                        IoU_table[i, j] = intersection / union
                    else:
                        IoU_table[i, j] = 0.0

            row_indices, col_indices = linear_sum_assignment(-IoU_table)
            matched_indices = np.array(
                list(zip(row_indices, col_indices))
            )  # list of (row, col) tuples
            unmatched_indices_tile = np.setdiff1d(
                np.arange(len(ids_tile_overlap)), row_indices
            )

            for m in matched_indices:
                if IoU_table[m[0], m[1]] >= 0.5:  # (tile, current)
                    # wherever the tile is m[0], it should be assigned m[1] in the larger prediction image
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = ids_current[m[1]]
                elif IoU_table[m[0], m[1]] == 0:
                    # there is no intersection
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = last
                    last += 1
                else:
                    # otherwise just take a union of the both ...
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = last
                    # instance_map_current[instance_map_current == ids_current[m[1]]] = last
                    last += 1
            for index in unmatched_indices_tile:  # not a tuple
                instance_map_current[
                    instance_map_tile == ids_tile_overlap[index]
                ] = last
                last += 1
            for id in ids_tile_notin_overlap:
                instance_map_current[instance_map_tile == id] = last
                last += 1

        return last, instance_map_current
    else:
        return (
            last,
            instance_map_current,
        )  # if there are no ids in tile, then just return


def predict_3d(
    im,
    model,
    tta,
    cluster_fast,
    n_sigma,
    fg_thresh,
    seed_thresh,
    min_mask_sum,
    min_unclustered_sum,
    min_object_size,
    cluster,
):
    """

    Parameters
    ----------
    im : PyTorch Tensor
        BZCYX

    model: PyTorch model

    tta: bool
        If True, then Test-Time Augmentation is on, otherwise off
    cluster_fast: bool
        If True, then the cluster.cluster() is used
        If False, then cluster.cluster_local_maxima() is used
    n_sigma: int
        This should be set equal to `3` for a 3D setting
    fg_thresh: float
        This should be set equal to `0.5` by default
    seed_thresh: float
        This should be set equal to `0.9` by default
    min_mask_sum: int
        Only start creating instances, if there are at least `min_mask_sum` pixels in foreground!
    min_unclustered_sum: int
        Stop when the number of seed candidates are less than `min_unclustered_sum`
    min_object_size: int
        Predicted Objects below this threshold are ignored

    cluster: Object of class `Cluster_3d`

    Returns
    -------
    instance_map: PyTorch Tensor
        ZYX
    seed_map: PyTorch Tensor
        ZYX
    """
    im, diff_x, diff_y, diff_z = pad_3d(im)

    if tta:
        for iter in tqdm(range(16), position=0, leave=True):
            if iter == 0:
                output_average = apply_tta_3d(im, model, iter)
            else:
                output_average = (
                    1
                    / (iter + 1)
                    * (output_average * iter + apply_tta_3d(im, model, iter))
                )  # iter
        output = torch.from_numpy(output_average).float().cuda()
    else:
        output = model(im)

    if cluster_fast:
        instance_map = cluster.cluster(
            output[0],
            n_sigma=n_sigma,
            fg_thresh=fg_thresh,
            seed_thresh=seed_thresh,
            min_mask_sum=min_mask_sum,
            min_unclustered_sum=min_unclustered_sum,
            min_object_size=min_object_size,
        )
    else:
        instance_map = cluster.cluster_local_maxima(
            output[0],
            n_sigma=n_sigma,
            fg_thresh=fg_thresh,
            min_mask_sum=min_mask_sum,
            min_unclustered_sum=min_unclustered_sum,
            min_object_size=min_object_size,
        )
    seed_map = torch.sigmoid(output[0, -1, ...])
    # unpad instance_map, seed_map

    if diff_z != 0:
        instance_map = instance_map[:-diff_z, :, :]
        seed_map = seed_map[:-diff_z, :, :]
    if diff_y != 0:
        instance_map = instance_map[:, :-diff_y, :]
        seed_map = seed_map[:, :-diff_y, :]
    if diff_x != 0:
        instance_map = instance_map[:, :, :-diff_x]
        seed_map = seed_map[:, :, :-diff_x]
    return instance_map, seed_map


def round_up_8(x):
    """Helper function for rounding integer to next multiple of 8

    e.g:
    round_up_8(10) = 16

        Parameters
        ----------
        x : int
            Integer
        Returns
        -------
        int
    """
    return (int(x) + 7) & (-8)


def pad_3d(im_tile):
    """Pad a 3D  image so that its dimensions are all multiples of 8

    Parameters
    ----------
    im_tile : numpy array (D x H x W)
       3D Image which needs to be padded!
    Returns
    -------
    (numpy array, int, int, int)
    (Padded 3D image, diff in x, diff in y, diff in z)
    The last three values are the amount of padding needed in the x, y and z dimensions
    """

    multiple_z = im_tile.shape[2] // 8
    multiple_y = im_tile.shape[3] // 8
    multiple_x = im_tile.shape[4] // 8
    if im_tile.shape[2] % 8 != 0:
        diff_z = 8 * (multiple_z + 1) - im_tile.shape[2]
    else:
        diff_z = 0
    if im_tile.shape[3] % 8 != 0:
        diff_y = 8 * (multiple_y + 1) - im_tile.shape[3]
    else:
        diff_y = 0
    if im_tile.shape[4] % 8 != 0:
        diff_x = 8 * (multiple_x + 1) - im_tile.shape[4]
    else:
        diff_x = 0

    p3d = (0, diff_x, 0, diff_y, 0, diff_z)  # last dim, second last dim, third last dim

    im_tile = F.pad(im_tile, p3d, "reflect")
    return im_tile, diff_x, diff_y, diff_z


def test_3d(fg_thresh, image, *args):
    """Infer the trained 3D model on 3D images

    Parameters
    ----------
    fg_thresh : float
        foreground threshold decides which pixels are considered for clustering, based on the predicted seediness scores at these pixels.
    args: dictionary
        Contains other paremeters such as `ap_val`, `seed_thresh` etc
    Returns
    -------
    float
        Average `AP_dsb` over all test images
    """
    (
        seed_thresh,
        ap_val,
        min_mask_sum,
        min_unclustered_sum,
        min_object_size,
        tta,
        model,
        save_images,
        save_results,
        save_dir,
        verbose,
        grid_x,
        grid_y,
        grid_z,
        pixel_x,
        pixel_y,
        pixel_z,
        one_hot,
        mask_region,
        n_sigma,
        cluster_fast,
        expand_grid,
        uniform_ds_factor,
    ) = args

    model.eval()
    # cluster module
    cluster = Cluster_3d(grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x)

    with torch.no_grad():
        im = image
        D, H, W = im.shape[2], im.shape[3], im.shape[4]

        if D > grid_z or H > grid_y or W > grid_x:
            if expand_grid:
                D_, H_, W_ = round_up_8(D), round_up_8(H), round_up_8(W)
                temp = np.maximum(H_, W_)
                H_ = temp
                W_ = temp
                pixel_x_modified = pixel_y_modified = H_ / grid_y
                pixel_z_modified = D_ * pixel_z / grid_z
                cluster = Cluster_3d(
                    D_, H_, W_, pixel_z_modified, pixel_y_modified, pixel_x_modified
                )
                instance_map, seed_map = predict_3d(
                    im,
                    model,
                    tta,
                    cluster_fast,
                    n_sigma,
                    fg_thresh,
                    seed_thresh,
                    min_mask_sum,
                    min_unclustered_sum,
                    min_object_size,
                    cluster,
                )
            else:
                # here, we try stitching predictions instead
                last = 1
                instance_map = np.zeros((D, H, W), dtype=np.int16)
                seed_map = np.zeros((D, H, W), dtype=np.float64)
                num_overlap_pixels = 4
                for z in range(0, D, grid_z - num_overlap_pixels):
                    for y in range(0, H, grid_y - num_overlap_pixels):
                        for x in range(0, W, grid_x - num_overlap_pixels):
                            instance_map_tile, seed_map_tile = predict_3d(
                                im[
                                    :, :, z : z + grid_z, y : y + grid_y, x : x + grid_x
                                ],
                                model,
                                tta,
                                cluster_fast,
                                n_sigma,
                                fg_thresh,
                                seed_thresh,
                                min_mask_sum,
                                min_unclustered_sum,
                                min_object_size,
                                cluster,
                            )
                            last, instance_map = stitch_3d(
                                instance_map_tile.cpu().detach().numpy(),
                                instance_map,
                                z,
                                y,
                                x,
                                last,
                                num_overlap_pixels,
                            )
                            seed_map[z : z + grid_z, y : y + grid_y, x : x + grid_x] = (
                                seed_map_tile.cpu().detach().numpy()
                            )
                instance_map = torch.from_numpy(instance_map).cuda()
                seed_map = torch.from_numpy(seed_map).float().cuda()
        else:
            instance_map, seed_map = predict_3d(
                im,
                model,
                tta,
                cluster_fast,
                n_sigma,
                fg_thresh,
                seed_thresh,
                min_mask_sum,
                min_unclustered_sum,
                min_object_size,
                cluster,
            )

        instance_map = instance_map.cpu().detach().numpy().astype(np.uint16)
        seed_map = seed_map.cpu().detach().numpy()

    return instance_map, seed_map
