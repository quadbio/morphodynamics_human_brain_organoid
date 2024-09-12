import cupy as cp
import cupyx as cpx
import matplotlib.animation as animation
import matplotlib.colorbar
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from cupyx.scipy.ndimage import gaussian_filter, rotate
from IPython.display import HTML, clear_output
from matplotlib.colors import ListedColormap
from skimage import exposure
from skimage.transform import rescale
from tifffile import imread, imsave

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

# implementaion of royerlab DEXP
def attenuation_filter(
    image, attenuation_min_density, attenuation, attenuation_filtering
):
    if attenuation_filtering > 0:
        image_for_attenuation = gaussian_filter(image, sigma=attenuation_filtering)
    else:
        image_for_attenuation = image

    cum_density = cp.cumsum(
        attenuation_min_density + (1 - attenuation_min_density) * image_for_attenuation,
        axis=0,
    )

    image *= cp.exp(-attenuation * cum_density)
    return image


def create_colored_image(im_proj, lower_percentile=0.5, upper_percentile=99.5):
    green_map = [[0, i / 255, 0] for i in range(256)]
    green_matplotlib_map = ListedColormap(green_map, "Green")
    magenta_map = [[i / 255, 0, i / 255] for i in range(256)]
    magenta_matplotlib_map = ListedColormap(magenta_map, "Magenta")
    im_proj = skimage.exposure.rescale_intensity(im_proj, out_range=(0, 1))
    vmin_green, vmax_green = np.percentile(
        im_proj[0, :, :], q=(lower_percentile, upper_percentile)
    )
    clipped_green = exposure.rescale_intensity(
        im_proj[0, :, :], in_range=(vmin_green, vmax_green), out_range=np.float32
    )

    vmin_magenta, vmax_magenta = np.percentile(
        im_proj[1, :, :], q=(lower_percentile, upper_percentile)
    )
    clipped_magenta = exposure.rescale_intensity(
        im_proj[1, :, :], in_range=(vmin_magenta, vmax_magenta), out_range=np.float32
    )

    channel1 = green_matplotlib_map(clipped_green)
    channel2 = magenta_matplotlib_map(clipped_magenta)
    assembled = np.stack((channel1, channel2), axis=3)
    newim = np.max(assembled, axis=3)
    return newim


def read_tiff_stacks(input_dir, time_point, scale_factor, pad_size=32):

    if scale_factor != [1.0, 1.0, 1.0]:
        stack_mcherry_downscaled = rescale(
            imread(input_dir + "mCherry/t" + f"{time_point:04}" + "_mCherry.tif"),
            scale_factor,
            anti_aliasing=True,
        )
        stack_mcherry_downscaled = np.pad(
            stack_mcherry_downscaled, ((pad_size, pad_size)), "constant"
        )

        stack_gfp_downscaled = rescale(
            imread(input_dir + "GFP/t" + f"{time_point:04}" + "_GFP.tif"),
            scale_factor,
            anti_aliasing=True,
        )
        stack_gfp_downscaled = np.pad(
            stack_gfp_downscaled, ((pad_size, pad_size)), "constant"
        )
    elif scale_factor == [1.0, 1.0, 1.0]:
        stack_mcherry_downscaled = imread(
            input_dir + "mCherry/t" + f"{time_point:04}" + "_mCherry.tif"
        )
        stack_gfp_downscaled = imread(
            input_dir + "GFP/t" + f"{time_point:04}" + "_GFP.tif"
        )

    return stack_gfp_downscaled, stack_mcherry_downscaled


def create_movie(
    input_dir,
    output_name,
    time_point_start=1,
    time_point_stop=1,
    start_angle=1,
    stop_angle=360,
    n_frames=213,
    scale=0.25,
    voxel_sizes=[2, 0.347, 0.347],
    attenuation=True,
    MIP=True,
    stack_slice=None,
    run_through_slice=False,
    rotation_axes=(0, 2),
    attenuation_filtering=4,
    attenuation_min_density=0.002,
    attenuation_strength=0.01,
    pad_size=32,
):

    scale_factor = [
        scale * (voxel_sizes[0] / np.min(voxel_sizes)),
        scale * (voxel_sizes[1] / np.min(voxel_sizes)),
        scale * (voxel_sizes[2] / np.min(voxel_sizes)),
    ]
    print(scale_factor)

    if time_point_start == time_point_stop:
        stack_gfp_downscaled, stack_mcherry_downscaled = read_tiff_stacks(
            input_dir, time_point_start, scale_factor, pad_size=pad_size
        )
        if run_through_slice == True:
            time_range = np.arange(0, len(stack_gfp_downscaled)).astype(int)
            angle_range = np.linspace(start_angle, stop_angle, len(time_range))

    if run_through_slice == False:
        angle_range = np.linspace(start_angle, stop_angle, n_frames)
        time_range = np.linspace(time_point_start, time_point_stop, n_frames).astype(
            int
        )

    print(angle_range)

    print(time_range)

    assert len(angle_range) == len(time_range)

    ims = []
    tp_old = -1
    fig, axes = plt.subplots(1, figsize=(20, 20))
    for angle, time_point in zip(angle_range, time_range):
        print(time_point)
        if time_point_start != time_point_stop:
            if time_point != tp_old:
                stack_gfp_downscaled, stack_mcherry_downscaled = read_tiff_stacks(
                    input_dir, time_point, scale_factor, pad_size=pad_size
                )
            print("loaded image")
        if start_angle != stop_angle:
            stack_mcherry_rotated = (
                np.nan_to_num(
                    rotate(
                        cp.asarray(stack_mcherry_downscaled),
                        angle,
                        mode="constant",
                        axes=rotation_axes,
                        reshape=False,
                    )
                )
                * 1000
            )
            stack_gfp_rotated = (
                np.nan_to_num(
                    rotate(
                        cp.asarray(stack_gfp_downscaled),
                        angle,
                        mode="constant",
                        axes=rotation_axes,
                        reshape=False,
                    )
                )
                * 1000
            )
            print("rotated image")
        else:
            image = np.stack([stack_gfp_downscaled, stack_mcherry_downscaled], axis=0)
            print("stacked images", image.shape)

        if attenuation == True:
            stack_mcherry_rotated = attenuation_filter(
                stack_mcherry_rotated,
                attenuation_min_density,
                attenuation_strength,
                attenuation_filtering,
            ).get()
            stack_gfp_rotated = attenuation_filter(
                stack_gfp_rotated,
                attenuation_min_density,
                attenuation_strength,
                attenuation_filtering,
            ).get()
            image = np.stack([stack_gfp_rotated, stack_mcherry_rotated], axis=0)
            print("attenuated image")
        if MIP == True:
            im_proj = np.max(image, axis=1)
        elif stack_slice != None:
            im_proj = image[:, stack_slice]
        elif run_through_slice == True:
            im_proj = image[:, time_point]
        print("image projection done")
        newim = create_colored_image(
            im_proj, lower_percentile=0.5, upper_percentile=99.5
        )
        a1 = axes.imshow(newim)
        ims.append([a1])
        axes.set_axis_off()

        tp_old = time_point

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    fig.tight_layout()
    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat=False, repeat_delay=0
    )
    clear_output()
    ani.save(output_name, bitrate=25000)
