import cv2
import h5py
import numpy as np
import skimage
from scipy import ndimage as ndi
from skimage.measure import label, regionprops, regionprops_table

# import h5py
from skimage.morphology import binary_closing, disk, remove_small_holes


def read_and_stack(time_point, file_path, slice_ind=(), rescale_stack=True):
    stack_file = h5py.File(file_path, "r")
    stack_combined = stack_file[f"t{time_point:05}"]["s00"]["0"]["cells"][
        slice_ind
    ].astype(np.float32)
    stack_combined += stack_file[f"t{time_point:05}"]["s01"]["0"]["cells"][
        slice_ind
    ].astype(np.float32)
    if rescale_stack == True:
        return rescale(stack_combined, 0.25, anti_aliasing=True, preserve_range=True)
    else:
        return stack_combined


def postprocess_masks(combined_masks):
    # Fill any holes in the lumen
    mask_lumen = combined_masks == 3
    mask_lumen_filled_holes = ndi.binary_fill_holes(mask_lumen)
    combined_masks[mask_lumen_filled_holes] = 3

    # Background mask
    mask_background = combined_masks == 1

    # Fill holes in organoid
    mask_all = combined_masks >= 2
    mask_organoid_filled_holes = ndi.binary_fill_holes(mask_all)

    # Background within organoid = lumen
    mask_background_within_organoid = mask_organoid_filled_holes * mask_background
    combined_masks[mask_background_within_organoid] = 3

    # Convex hull of organoid
    mask_organoid = combined_masks == 2
    mask_organoid_convex_hull = skimage.morphology.convex_hull_object(mask_organoid)

    # Lumen outside of the convex hull of the organoid==background
    mask_lumen = combined_masks == 3
    masked_lumen_outside_organoid = ~mask_organoid_convex_hull * mask_lumen
    combined_masks[masked_lumen_outside_organoid] = 1
    return combined_masks


def keep_largest_object(combined_masks):
    # largest object is the organoid, rest is background
    organoid_mask = combined_masks > 1
    organoid_mask_label = label(organoid_mask)
    region_size = []
    for region in regionprops(organoid_mask_label):
        # take regions with large enough areas
        region_size.append(region.area)

    max_size = np.where(region_size == max(region_size))[0][0] + 1
    organoid_mask = organoid_mask_label != max_size
    combined_masks[organoid_mask] = 1

    return combined_masks


def smooth_organoids(combined_masks, sigma):
    mask_organoid = combined_masks == 2
    mask_lumen = combined_masks == 3
    if len(combined_masks.shape) > 2:
        # Smooth using an anisotropic gaussian
        mask_lumen_smoothed = skimage.filters.gaussian(
            mask_lumen.astype(float), sigma=[sigma * (0.347 * 4) / 2, sigma, sigma]
        )
        mask_lumen_smoothed = mask_lumen_smoothed > 0.5

        mask_organoid_smoothed = skimage.filters.gaussian(
            mask_organoid.astype(float), sigma=[sigma * (0.347 * 4) / 2, sigma, sigma]
        )
        mask_organoid_smoothed = mask_organoid_smoothed > 0.5

    elif len(combined_masks.shape) == 2:
        # Smooth using an gaussian
        mask_lumen_smoothed = skimage.filters.gaussian(
            mask_lumen.astype(float), sigma=sigma
        )
        mask_lumen_smoothed = mask_lumen_smoothed > 0.5

        mask_organoid_smoothed = skimage.filters.gaussian(
            mask_organoid.astype(float), sigma=sigma
        )
        mask_organoid_smoothed = mask_organoid_smoothed > 0.5

    # Put together new smoothed tissue mask
    combined_masks = np.ones(combined_masks.shape)
    combined_masks[mask_organoid_smoothed] = 2
    combined_masks[mask_lumen_smoothed] = 3
    return combined_masks
