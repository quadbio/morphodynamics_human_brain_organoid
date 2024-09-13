import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import h5py
import jetraw
import numpy as np
import pandas as pd
import zarr
from joblib import Parallel, delayed
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from skimage.io import imread
from skimage.transform import rescale
from tqdm import tqdm


def write_time_point(
    time_point: str,
    channel_group: str,
    chunk_size: tuple,
    channel_h5: str,
    h5_file: str,
    downscale=None,
):
    """
    Convert a single channel image stored in an HDF5 file to the OME-ZARR format.

    Parameters:
    - time_point (str): The time point identifier.
    - channel_group (h5py.Group): The HDF5 group representing the channel.
    - chunk_size (tuple): Tuple specifying the chunk size for storage.
    - channel_h5 (str): The name of the channel in the HDF5 file.
    - downscale (float, optional): Downsampling factor. Default is None.

    Returns:
    - None

    Notes:
    - This function assumes that `stack_file` is a global variable representing
      the HDF5 file containing the image stack.
    - The image data is stored in the provided `channel_group` under the
      specified `time_point`.
    - If `downscale` is not None, the image is downsampled before writing.

    """
    time_group = channel_group.require_group(time_point)
    stack_file = h5py.File(h5_file, "r")
    image = np.array(
        stack_file[f"t{int(time_point)+1:05}"][channel_h5]["0"]["cells"][:]
    )
    if downscale != None:
        image = (
            rescale(image, downscale, anti_aliasing=False, preserve_range=True)
            .clip(0.0, 2**16 - 1)
            .astype(np.uint16)
        )
    write_image(
        image=image,
        group=time_group,
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )


def write_time_point_multimosaic(
    time_point: str,
    channel_group: str,
    chunk_size: tuple,
    channel_h5: str,
    h5_file: str,
    downscale=None,
):
    """
    Convert a single channel image stored in an HDF5 file to the OME-ZARR format.

    Parameters:
    - time_point (str): The time point identifier.
    - channel_group (h5py.Group): The HDF5 group representing the channel.
    - chunk_size (tuple): Tuple specifying the chunk size for storage.
    - channel_h5 (str): The name of the channel in the HDF5 file.
    - downscale (float, optional): Downsampling factor. Default is None.

    Returns:
    - None

    Notes:
    - This function assumes that `stack_file` is a global variable representing
      the HDF5 file containing the image stack.
    - The image data is stored in the provided `channel_group` under the
      specified `time_point`.
    - If `downscale` is not None, the image is downsampled before writing.

    """
    time_group = channel_group.require_group(time_point)
    stack_file = h5py.File(h5_file, "r")
    image = np.array(
        stack_file[f"t{(int(time_point)*2+1):05}"][channel_h5]["0"]["cells"][:]
    )
    if downscale != None:
        image = (
            rescale(image, downscale, anti_aliasing=False, preserve_range=True)
            .clip(0.0, 2**16 - 1)
            .astype(np.uint16)
        )
    write_image(
        image=image,
        group=time_group,
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )


def h5_to_omezarr_multimosaic(
    h5_file,
    out_path,
    channel_names,
    h5_channel_names,
    time_points=None,
    n_jobs=8,
    downscale=None,
):

    """
    Convert a multi-channel image stack stored in an HDF5 file to the OME-ZARR format.

    Parameters:
    - h5_file (str): Path to the input HDF5 file.
    - out_path (str): Path to the output OME-ZARR file.
    - channel_names (list of str): List of channel names for the OME-ZARR file.
    - h5_channel_names (list of str): List of corresponding channel names in the HDF5 file.
    - time_points (int or None, optional): Number of time points to process. Default is None,
      meaning all time points in the HDF5 file are processed.
    - n_jobs (int, optional): Number of parallel jobs to run. Default is 8.
    - downscale (float, optional): Downsampling factor for image resizing. Default is None.

    Returns:
    - None

    Notes:
    - This function converts a multi-channel image stack stored in an HDF5 file to the OME-ZARR format.
    - If `time_points` is None, all time points in the HDF5 file are processed.
    - `channel_names` and `h5_channel_names` should have the same length, where each element
      corresponds to the channel name in the OME-ZARR file and the corresponding channel name
      in the HDF5 file, respectively.
    - The output OME-ZARR file will have the same spatial dimensions as the input image stack.

    """

    stack_file = h5py.File(h5_file, "r")
    if time_points != None:
        time_points = np.arange(time_points).astype(str)
    else:
        time_points = np.arange(len(list(stack_file.keys())) - 2)

    input_shape = stack_file[f"t{1:05}"]["s00"]["0"]["cells"][:].shape
    size_z = input_shape[0]
    size_x = input_shape[1]
    size_y = input_shape[2]

    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)
    for channel_name, channel_h5 in zip(channel_names, h5_channel_names):
        channel_group = root.require_group(channel_name)
        chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)
        result = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=5)(
            delayed(write_time_point_multimosaic)(
                time_point,
                channel_group,
                chunk_size,
                channel_h5,
                h5_file,
                downscale=downscale,
            )
            for time_point in time_points
        )


def h5_to_omezarr(
    h5_file,
    out_path,
    channel_names,
    h5_channel_names,
    time_points=None,
    n_jobs=8,
    downscale=None,
):

    """
    Convert a multi-channel image stack stored in an HDF5 file to the OME-ZARR format.

    Parameters:
    - h5_file (str): Path to the input HDF5 file.
    - out_path (str): Path to the output OME-ZARR file.
    - channel_names (list of str): List of channel names for the OME-ZARR file.
    - h5_channel_names (list of str): List of corresponding channel names in the HDF5 file.
    - time_points (int or None, optional): Number of time points to process. Default is None,
      meaning all time points in the HDF5 file are processed.
    - n_jobs (int, optional): Number of parallel jobs to run. Default is 8.
    - downscale (float, optional): Downsampling factor for image resizing. Default is None.

    Returns:
    - None

    Notes:
    - This function converts a multi-channel image stack stored in an HDF5 file to the OME-ZARR format.
    - If `time_points` is None, all time points in the HDF5 file are processed.
    - `channel_names` and `h5_channel_names` should have the same length, where each element
      corresponds to the channel name in the OME-ZARR file and the corresponding channel name
      in the HDF5 file, respectively.
    - The output OME-ZARR file will have the same spatial dimensions as the input image stack.

    """

    stack_file = h5py.File(h5_file, "r")
    if time_points != None:
        time_points = np.arange(time_points).astype(str)
    else:
        time_points = np.arange(len(list(stack_file.keys())) - 2)

    input_shape = stack_file[f"t{1:05}"]["s00"]["0"]["cells"][:].shape
    size_z = input_shape[0]
    size_x = input_shape[1]
    size_y = input_shape[2]

    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)
    for channel_name, channel_h5 in zip(channel_names, h5_channel_names):
        channel_group = root.require_group(channel_name)
        chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)
        result = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=5)(
            delayed(write_time_point)(
                time_point,
                channel_group,
                chunk_size,
                channel_h5,
                h5_file,
                downscale=downscale,
            )
            for time_point in time_points
        )


def parse_xml_to_list(xml_content):
    namespaces = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    tree = ET.parse(xml_content)
    root = tree.getroot()

    # Extract SizeT, SizeX, SizeY, SizeZ from the Pixels element
    pixels = root.find(".//ome:Pixels", namespaces)
    size_t = pixels.attrib["SizeT"]
    size_x = pixels.attrib["SizeX"]
    size_y = pixels.attrib["SizeY"]
    size_z = pixels.attrib["SizeZ"]

    # Extract file names and FirstT values from TiffData elements
    tiff_data_list = []
    for tiff_data in pixels.findall("ome:TiffData", namespaces):
        first_t = tiff_data.attrib["FirstT"]
        file_name = tiff_data.find("ome:UUID", namespaces).attrib["FileName"]
        tiff_data_list.append([file_name, size_t, size_x, size_y, size_z, first_t])
    column_names = ["file_name", "n_time_points", "n_x", "n_y", "n_z", "time_point"]
    result = pd.DataFrame(tiff_data_list, columns=column_names)
    result["channel"] = (
        result["file_name"]
        .str.split(".", n=1, expand=True)[0]
        .str.split("_", n=1, expand=True)[1]
    )
    return result


def write_time_point_tiff(
    time_point: str,
    channel_group: str,
    chunk_size: tuple,
    ome_df_channel,
    downscale=None,
):
    """
    Convert a single channel image stored in OME tiff to the OME-ZARR format.

    """
    time_group = channel_group.require_group(time_point)

    ome_df_tp = ome_df_channel[ome_df_channel["time_point"] == str(time_point)]
    try:
        image = jetraw.imread(ome_df_tp["file_name"].iloc[0])
    except:
        image = imread(ome_df_tp["file_name"].iloc[0])

    if downscale != None:
        image = (
            rescale(image, downscale, anti_aliasing=False, preserve_range=True)
            .clip(0.0, 2**16 - 1)
            .astype(np.uint16)
        )
    write_image(
        image=image,
        group=time_group,
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )


def tiff_to_omezarr(
    input_path,
    out_path,
    n_jobs=8,
    downscale=None,
):

    """
    Convert a multi-channel image stack stored in an OME-Tiff to the OME-ZARR format.


    """

    omes = glob(input_path + "/*.ome", recursive=True)
    ome_path = Path(omes[0])
    ome_df = parse_xml_to_list(ome_path)
    ome_df["file_name"] = input_path + ome_df["file_name"]
    input_shape = ome_df.iloc[0]
    size_z = int(input_shape["n_z"])
    size_x = int(input_shape["n_x"])
    size_y = int(input_shape["n_y"])
    time_points = np.unique(ome_df["time_point"].astype(int))
    channel_names = np.unique(ome_df["channel"])

    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)

    for channel_name in channel_names:
        ome_df_channel = ome_df[ome_df["channel"] == channel_name]
        channel_group = root.require_group(channel_name)
        chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)
        result = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=20)(
            delayed(write_time_point_tiff)(
                time_point,
                channel_group,
                chunk_size,
                downscale=downscale,
                ome_df_channel=ome_df_channel,
            )
            for time_point in time_points
        )
