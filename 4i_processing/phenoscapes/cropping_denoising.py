from skimage import io
from pathlib import Path
from phenoscapes.utils import get_metadata, crop_image, denoise_image, convert_uint16
import numpy as np
from functools import partial
from multiprocessing import Pool
# TODO: decouple cropping and denoising
from smo import SMO
from skimage.filters import median
from skimage.morphology import disk

def smo_background_subtraction(img,
                               sigma=0,
                               size=7,
                               shape=(1024, 1024),
                               mask_zeros=False,
                               sample_factor=None):
    """
    Smo background of a given sample.
    :param img:
    :param sigma:
    :param size:
    :param shape:
    :return: background subtracted image
    """
    if mask_zeros:
        # Mask largest value + zeros
        img = img.astype(float)
        #masked_image = np.ma.masked_greater_equal(img, img.max()) -> smo should mask for max values by it self
        masked_image = np.ma.masked_less_equal(img, 0)
        # get background
        smo = SMO(sigma=sigma, size=size, shape=shape)
        bg_mask = smo.bg_mask(masked_image, threshold=0.05)
        # get bg_value
        bg_value = np.median(bg_mask.compressed())
        img -= bg_value
    else:
        smo = SMO(sigma=sigma, size=size, shape=shape)
        img = smo.bg_corrected(img)
        img[img < 0] = 0
    return img

def process_image(file: str, dir_images: str, dir_output: str,
                  mask: np.array, mask_cropped: np.array,
                  crop: bool = True,
                  run_smo: bool = False,
                  denoise: bool = True,
                 mask_zeros_smo:bool=False):
    """
    Process a single image.
    :param denoise:
    :param crop:
    :param run_smo:
    :param file:
    :param dir_images:
    :param dir_output:
    :param mask:
    :param mask_cropped:
    :return:
    """
    img = io.imread(Path(dir_images, file))
    if run_smo:
        img = smo_background_subtraction(img.astype(float),mask_zeros=mask_zeros_smo)
    if crop:
        img = crop_image(img, mask)
        img = img * mask_cropped
    if denoise:
        img = denoise_image(img)
    img = convert_uint16(img).astype('uint16')
    io.imsave(Path(dir_output, file), img, check_contrast=False)


def run_cropping_and_denoising(sample,
                               dir_input: str,
                               dir_masks: str,
                               dir_output: str,
                               dir_output_masks: str,
                               dir_avg_nuclei: str,
                               config: dict = None):
    """
    Run cropping and denoising on a given sample.
    :param dir_avg_nuclei:
    :param config:
    :param sample:
    :param dir_input:
    :param dir_masks:
    :param dir_output:
    :param dir_output_masks:
    :param run_smo:
    :param dir_avg_nuclei:
    :return:
    """

    # Get metadata, set paths and create directories
    dir_images = Path(dir_input, sample)
    df = get_metadata(dir_images, custom_regex=config['regex'])
    dir_output = Path(dir_output, sample)
    dir_output.mkdir(parents=True, exist_ok=True)
    Path(dir_output_masks).mkdir(parents=True, exist_ok=True)

    # Get mask path, load, crop, save mask
    mask = io.imread(Path(dir_masks, sample + '.tif'))
    mask_cropped = crop_image(mask, mask).astype('uint8')
    io.imsave(Path(dir_output_masks, sample + '.tif'), mask_cropped, check_contrast=False)

    # handle nuclei averaging
    if config['cropping_denoising']['average_nuclei']:
        print('Averaging nuclei...')
        df_nuclei = df[df['channel_id'] == config['channel_nuclei']]
        files_nuclei = df_nuclei['file'].values

        if config['cropping_denoising']['smo']:
            nuclei = [io.imread(Path(dir_images, file)) for file in files_nuclei]
            nuclei = np.asarray([smo_background_subtraction(img) for img in nuclei])
        else:
            nuclei = np.asarray([io.imread(Path(dir_images, file)) for file in files_nuclei])
        nuclei = np.mean(nuclei, axis=0)
        nuclei = nuclei.astype('uint16')
        # median filter for removal of hot pixels
        nuclei = median(nuclei, disk(1))
        if config['cropping_denoising']['denoising']:
            nuclei = denoise_image(nuclei)
        if config['cropping_denoising']['cropping']:
            nuclei = crop_image(nuclei, mask)
            nuclei = nuclei * mask_cropped
        Path(dir_avg_nuclei).mkdir(parents=True, exist_ok=True)
        io.imsave(Path(dir_avg_nuclei, f'{sample}.tif'), nuclei, check_contrast=False)

    # Get file names of reference dapi and all other channels/cycles
    file_nuclei = df[(df['cycle_id'] == config['reference_cycle']) & (df['channel_id'] == config['channel_nuclei'])][
        'file'].values
    df = df[df['channel_id'].isin(config['channel_stains'])]
    files_images = df['file'].values
    files_images = np.concatenate([file_nuclei, files_images])

    # Read, crop, mask, denoise and save images
    process_image_partial = partial(process_image,
                                    dir_images=dir_images,
                                    dir_output=dir_output,
                                    mask=mask,
                                    mask_cropped=mask_cropped,
                                    crop=config['cropping_denoising']['cropping'],
                                    run_smo=config['cropping_denoising']['smo'],
                                    mask_zeros_smo=config['cropping_denoising']['mask_zeros_smo'],
                                    denoise=config['cropping_denoising']['denoising'])
    if config['cropping_denoising']['multiprocessing']:
        n_processes = config['cropping_denoising']['n_processes']
        with Pool(n_processes) as p:
            p.map(process_image_partial, files_images)
    else:
        for file in files_images:
            process_image_partial(file)
