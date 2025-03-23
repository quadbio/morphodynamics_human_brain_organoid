from phenoscapes.utils import get_metadata, scale_image
from cellpose import models
from skimage import io
from skimage.color import label2rgb
from pathlib import Path
import numpy as np
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

def run_cellpose(sample:str,
                 dir_images:str,
                 dir_output:str,
                 dir_avg_nuclei:str = None,
                 diameter:int = 20,
                 model_type:str = 'cyto2',
                 flow_threshold:int = -3,
                 cellprob_threshold:int = 1,
                 config:dict = None,
                 pretrained_model:str=False):
    """
    Run CellPose on a given sample.
    :param sample:
    :param dir_images:
    :param dir_output:
    :param dir_avg_nuclei:
    :param diameter:
    :param model_type:
    :param flow_threshold:
    :param cellprob_threshold:
    :param config:
    :param pretrained_model:
    :return:
    """
    print('Running CellPose.')
    
    if config['cropping_denoising']['average_nuclei']:
        dir_images_sample = Path(dir_avg_nuclei)
        file = f'{sample}.tif'
    elif config['cellpose']['segment_channel']!=False:
        dir_images_sample = Path(dir_images, sample)
        df_images = get_metadata(dir_images_sample, custom_regex=config['regex'])
        file = df_images[(df_images['channel_id'] == config['cellpose']['segment_channel']) &
                         (df_images['cycle_id'] == config['cellpose']['segment_cycle'])]['file'].values[0]
    else:
        dir_images_sample = Path(dir_images, sample)
        df_images = get_metadata(dir_images_sample, custom_regex=config['regex'])
        file = df_images[(df_images['channel_id'] == config['channel_nuclei']) &
                         (df_images['cycle_id'] == config['reference_cycle'])]['file'].values[0]
        
    if pretrained_model!=False:
        model = models.CellposeModel(gpu=False,pretrained_model=pretrained_model)

        img = io.imread(Path(dir_images_sample, file))
        mask, flows, styles = model.eval(img, diameter=diameter, channels=[[0, 0]], do_3D=False,
                                                flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)

    else:
        model = models.Cellpose(gpu=False, model_type=model_type)

        img = io.imread(Path(dir_images_sample, file))
        mask, flows, styles, diams = model.eval(img, diameter=diameter, channels=[[0, 0]], do_3D=False,
                                                flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
    
    Path(dir_output).mkdir(exist_ok=True, parents=True)
    io.imsave(Path(dir_output, sample + '.tif'), mask, check_contrast=False)

def rescale_image(img, min_quant=0, max_quant=0.99):
    # turn type to float before rescaling
    img = img * 1.0
    min_val = np.quantile(img, min_quant)
    max_val = np.quantile(img, max_quant)
    img = rescale_intensity(img, in_range=(min_val, max_val))
    return img

def combine_vertically(img1, img2, border=15):
    width = img1.shape[1]
    barrier = np.zeros((border, width), np.uint16)
    img = np.append(img1, barrier, axis=0)
    img = np.append(img, img2, axis=0)
    return img

def combine_horizontally(img1, img2, border=15):
    height = img1.shape[0]
    barrier = np.zeros((height, border), np.uint16)
    img = np.append(img1, barrier, axis=1)
    img = np.append(img, img2, axis=1)
    return img

def colour_nuclei(nuclei):
    coloured = np.zeros((nuclei.shape[0], nuclei.shape[1], 3), np.uint8)
    for n in range(nuclei.max()):
        pixels = (nuclei == n+1)
        coloured[pixels, :] = np.random.randint(1,255,3)
    # Add alpha channel to make background transparent
    alpha = np.all(coloured != 0, axis=2) * 255
    rgba = np.dstack((coloured, alpha)).astype(np.uint8)
    return rgba

def run_cellpose_sweep(sample:str,diam:int, model_type:str, dir_output:str, dir_images:str):
    dir_images = Path(dir_images,sample)
    df_images = get_metadata(dir_images)
    file = df_images[df_images['channel_id'] == 0]['file'].values[0]
    img_hoechst = io.imread(Path(dir_images,file))
    # Define cellpose model
    model = models.Cellpose(gpu=False, model_type=model_type)

    # Define two regions of interest
    y_min1 = 2000
    y_max1 = 2500
    x_min1 = 2000
    x_max1 = 2500

    y_min2 = 2000
    y_max2 = 2500
    x_min2 = 1000
    x_max2 = 1500

    # Combine ROIs into one image
    img1 = img_hoechst[y_min1:y_max1, x_min1:x_max1]
    img2 = img_hoechst[y_min2:y_max2, x_min2:x_max2]
    img_combined = combine_vertically(img1, img2)

    flow_thrs = np.linspace(0.6, 1, 3)
    cellprob_thrs = np.linspace(-1, -4, 4)

    channels = [0, 0]  # [0, 0] for grayscale
    nuclei_masks = []
    for flow_thr in flow_thrs:
        # Gather all images with the current flow_thr into a list
        subset = []
        for cellprob_thr in cellprob_thrs:
            mask, flows, styles, diams = model.eval(img_combined, diameter=diam, channels=channels, do_3D=False,
                                             flow_threshold=flow_thr, cellprob_threshold=cellprob_thr)
            subset.append(mask)
        # Append list of all images with current flow_thr to list of lists
        nuclei_masks.append(subset)


    # plotting
    fig, axes = plt.subplots(len(flow_thrs), len(cellprob_thrs), figsize=(30,30))

    # Turn off axis
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Add row labels
    for ax, thr in zip(axes[:, 0], flow_thrs):
        ax.set_ylabel("{}".format(np.round(thr, 2)), size=20)

    # Add column labels
    for ax, thr in zip(axes[0], cellprob_thrs):
        ax.set_title("{}".format(thr), size=20)

    for i in range(len(flow_thrs)):
        for j in range(len(cellprob_thrs)):
            # Colour the nuclei
            nuclei_coloured = colour_nuclei(nuclei_masks[i][j])
            # Plot segmentation on top of Hoechst image
            axes[i, j].imshow(rescale_image(img_combined), cmap="gray", clim=[0, 1.2])
            axes[i, j].imshow(nuclei_coloured, alpha=0.4)

    # Add "title" to signify what columns are
    plt.suptitle("Cell probability threshold", fontsize=25, y=0.98)

    # Create "outer" image in order to add a common y-label
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Flow threshold", fontsize=25)
    # Add bounding box to tight_layout because suptitle is ignored and will therefore overlap with plot otherwise
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure

    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(dir_output,"CellPose_threshold-matrix_type-{}_diam-{}.png".format(model_type, diam)))
    plt.close('all')

def generate_label_overlay(sample,
                         dir_images:str,
                         dir_segmented:str,
                         dir_output:str,
                         n_sampling:int = 2,
                         config:dict = None):

    dir_images_sample = Path(dir_images, sample)
    df_images = get_metadata(dir_images_sample, custom_regex=config['regex'])
    
    if config['cellpose']['segment_channel']!=False:
        file = df_images[(df_images['channel_id'] == config['cellpose']['segment_channel']) &
                         (df_images['cycle_id'] == config['cellpose']['segment_cycle'])]['file'].values[0]
    else:
        file = df_images[df_images['channel_id'] == config['channel_nuclei']]['file'].values[0]
    img = io.imread(Path(dir_images_sample, file))
    img = scale_image(img.astype('float'),range=(0,1))
    img_label = io.imread(Path(dir_segmented,sample+'.tif'))
    img_overlay = label2rgb(img_label, image=img, bg_label=0)
    # convert to uint8
    img_overlay = (img_overlay * 255).astype('uint8')
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    io.imsave(Path(dir_output,sample+'.png'),img_overlay[::n_sampling,::n_sampling,:], check_contrast=False)
