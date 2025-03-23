import itk
import numpy as np
import pandas as pd
from scipy import ndimage
from time import gmtime, strftime
from skimage import io
from matplotlib import pyplot as plt
from phenoscapes.utils import overlay_channels, scale_image, get_metadata, convert_uint16
from tqdm import tqdm
from pathlib import Path

def run_elastix(sample: str,
                dir_input: str,
                dir_output: str,
                config: dict,
                dir_masks: str = None):
    """
     Running Elastix and Transformix for dataset of one organoid and saves them to output directory
    :param sample: (int) id of organoid
    :param ref_cycles: (list[int]) cycles which are used as reference cycles
    :param dir_masks: (str) path to directory of initial simple masks
    :param dir_output: (str) path to output directory
    :param dir_input: (str) path to input directory
    :param config: (dict) parameter dictionary
    :return: None
    """

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print('Started alignment.')

    # Get metadata, set paths and create directories
    dir_images = Path(dir_input, sample)
    df = get_metadata(dir_images, custom_regex=config['regex'])
    dir_output = Path(dir_output, sample)
    dir_output.mkdir(parents=True, exist_ok=True)

    # Set reference cycle
    ref_cycle = config['reference_cycle']
    ch_nuclei = config['channel_nuclei']
    print('Set initial reference cycle to:', ref_cycle)

    # Arrange other cycles
    cycles = df['cycle_id'].unique().tolist()
    cycles.sort()
    cycles.remove(ref_cycle)

    # Load images and safe initial reference cycle
    images_ref = df[df['cycle_id'] == ref_cycle]['file'].values
    for image in images_ref:
        img = io.imread(Path(dir_images, image))
        io.imsave(Path(dir_output, image), img, check_contrast=False)

    # Load initial fixed image
    file_fixed_img = df[(df['cycle_id'] == ref_cycle) & (df['channel_id'] == ch_nuclei)]['file'].values[0]
    fixed_img = io.imread(Path(dir_images, file_fixed_img))
    print('Loaded initial fixed image.')

    # Load fixed mask
    if config['alignment']['mask']:
        fixed_mask = itk.imread(str(Path(dir_masks, sample + '.tif')), itk.UC)
        print('Loaded fixed mask.')
    else:
        fixed_mask = None

    # Initialize Elastix
    parameter_object = itk.ParameterObject.New()
    if config['alignment']['param_maps']['rigid'] is not None and config['alignment']['param_maps']['affine'] is not None:
        parameter_object.AddParameterFile(config['alignment']['param_maps']['rigid'])
        parameter_object.AddParameterFile(config['alignment']['param_maps']['affine'])

    else:
        parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
        parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')
        parameter_object.AddParameterMap(parameter_map_rigid)
        parameter_object.AddParameterMap(parameter_map_affine)
    print('Initialized Elastix.')

    # Iteratively apply alignment across all cycles
    for cycle in cycles:
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print('Processing cycle', cycle, '...')
        # Load moving image

        file_moving_img = df[(df['cycle_id'] == cycle) & (df['channel_id'] == ch_nuclei)]['file'].values[0]
        moving_img = io.imread(Path(dir_images, file_moving_img))
        print('Loaded moving image.')
        if config['alignment']['mask']:

            moving_mask = moving_img.copy()
            moving_mask = np.where(moving_mask >= 1, 1, 0)
            moving_mask = ndimage.binary_fill_holes(moving_mask).astype(np.uint16)
            MaskImageType = itk.Image[itk.UC, 2]
            moving_mask = itk.binary_threshold_image_filter(itk.GetImageFromArray(moving_mask),
                                                            lower_threshold=1,
                                                            inside_value=1,
                                                            ttype=(
                                                            type(itk.GetImageFromArray(moving_img)), MaskImageType))
            print('Loaded moving mask.')
        else:
            moving_mask = None

        # Call registration function
        print('Running Elastix.')
        moving_img, result_transform_parameters = itk.elastix_registration_method(
            itk.GetImageFromArray(fixed_img), itk.GetImageFromArray(moving_img),
            moving_mask=moving_mask, fixed_mask=fixed_mask,
            parameter_object=parameter_object)

        # Transformix
        print('Running Transformix.')
        channels = df[df['cycle_id'] == cycle]['channel_id'].unique().tolist()
        for channel in channels:
            file_img = df[(df['cycle_id'] == cycle) & (df['channel_id'] == channel)]['file'].values[0]
            if channel == ch_nuclei:
                img_aligned = itk.GetArrayFromImage(moving_img)
            else:
                img = io.imread(Path(dir_images, file_img)).astype(float)
                img_aligned = itk.transformix_filter(itk.GetImageFromArray(img), result_transform_parameters)
                img_aligned = itk.GetArrayFromImage(img_aligned)
            img_aligned = convert_uint16(img_aligned)
            io.imsave(Path(dir_output, file_img), img_aligned, check_contrast=False)

            if config['alignment']['reference_cycles'] is not None:
                if channel == ch_nuclei:
                    # Set new reference cycle if necessary
                    if cycle in config['alignment']['reference_cycles']:
                        ref_cycle = cycle
                        print('Setting new reference to:', ref_cycle)
                        fixed_img = img_aligned.copy()
                        if config['alignment']['mask']:
                            new_mask = np.where(img_aligned >= 1, 1, 0)
                            new_mask = ndimage.binary_fill_holes(new_mask).astype(int)
                            fixed_mask = fixed_mask * new_mask
                            fixed_mask = itk.binary_threshold_image_filter(
                                itk.GetImageFromArray(fixed_mask.astype(np.uint16)),
                                lower_threshold=1,
                                inside_value=1,
                                ttype=(type(itk.GetImageFromArray(fixed_img)), MaskImageType))
                            print('Updated fixed image and mask.')
        print('Finished alignment and saved aligned images.')


def check_alignment(sample: str, ch_nuclei: int, cycles: list,
                    regex: str,
                    dir_input: str,
                    dir_output: str,
                    show_plot: bool = False,
                    save_plot: bool = True,
                    n_sample: int = None) -> object:
    """

    :param ch_nuclei:
    :param sample:
    :param cycles:
    :param regex:
    :param dir_input:
    :param dir_output:
    :param show_plot:
    :param save_plot:
    :param n_sample:
    :return:
    """

    dir_images = Path(dir_input, sample)
    df = get_metadata(dir_images, custom_regex=regex)

    paths_imgs = [Path(dir_images, df[(df['cycle_id'] == cycle) & (df['channel_id'] == ch_nuclei)]['file'].values[0]) for cycle
                  in cycles]
    img_1 = io.imread(paths_imgs[0])
    img_2 = io.imread(paths_imgs[1])
    if len(cycles) == 2:
        overlay = overlay_channels(scale_image(img_1), scale_image(img_2))
    else:
        img_3 = io.imread(paths_imgs[2])
        overlay = overlay_channels(scale_image(img_1), scale_image(img_2), scale_image(img_3))
    if n_sample is not None:
        overlay = overlay[::n_sample, ::n_sample, :]
    # plotting
    plt.figure(figsize=(20, 20))
    plt.imshow(overlay)
    plt.suptitle(str(cycles), size=20)
    if save_plot:
        if dir_output is None:
            print('Specify dir_output in order to save plot.')
        else:
            Path(dir_output, sample).mkdir(parents=True, exist_ok=True)
            cycles = [str(cycle) for cycle in cycles]
            plt.savefig(Path(dir_output, sample, '_'.join(cycles) + '.png'))
            plt.close('all')
    if show_plot:
        return plt.show()


def run_checks(sample: str,
               dir_input: str,
               dir_output: str,
               df_stains: pd.DataFrame,
               config: dict):
    """
    :param config:
    :param df_stains:
    :param sample:
    :param dir_input:
    :param dir_output:
    :return:
    """
    ch_nuclei = config['channel_nuclei']
    # get number of cycles from df_stain
    cycles = df_stains['cycle_id'].unique().tolist()
    cycles.sort()
    # remove reference cycle
    cycles.remove(config['reference_cycle'])

    # generate list of lists with every second cycle
    cycles = [cycles[i:i + 2] for i in range(0, len(cycles), 2)]
    # add reference cycle to every list as first item
    cycles = [[config['reference_cycle']] + cycle for cycle in cycles]

    # generate dict with checks from cycles
    checks = {}
    for i in range(len(cycles)):
        checks[i] = cycles[i]

    if Path(dir_input, sample).is_dir():
        print('Processing sample:', sample)
        for i in tqdm(checks):
            check_alignment(sample=sample,
                            ch_nuclei=ch_nuclei,
                            regex=config['regex'],
                            cycles=checks[i],
                            dir_input=dir_input,
                            dir_output=dir_output,
                            show_plot=False,
                            save_plot=True,
                            n_sample=2)
