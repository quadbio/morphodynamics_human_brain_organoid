from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import yaml
from matplotlib import pyplot as plt
from phenoscapes.cli import load_default_config, process_sample
from phenoscapes.feature_extraction import extract_features
from phenoscapes.morphometrics import run_extract_morphology_features
from phenoscapes.sc import convert_to_h5ad, plot_summary
from phenoscapes.segmentation import generate_label_overlay, run_cellpose
from phenoscapes.utils import get_metadata
from skimage import io
from skimage.measure import regionprops
from tqdm import tqdm


def run_comaprtement_analysis(sample, default_config, iou_threshold=0.2):
    dir_segmented_bcat = Path(dir_output, "segmented_bcat")
    dir_avg_nuclei = Path(dir_output, "avg_nuclei")
    config = default_config
    config["cellpose"]["segment_channel"] = 2
    config["cellpose"]["segment_cycle"] = 1
    config["cellpose"][
        "pretrained_model"
    ] = "/cluster/scratch/gutgi/4i/Brain_ECM_4i_2_v2/cellpose_model/CP_20240707_cyto_lr_0_1_weight_decay_0_0001_epochs_300_bcat_v2"
    config["cellpose"]["cellprob_threshold"] = 0.0
    config["cellpose"]["flow_threshold"] = 0.9
    config["cellpose"]["segment_dual_channel"] = True
    run_cellpose(
        sample=sample,
        dir_images=dir_bg_subtracted,
        dir_output=dir_segmented_bcat,
        dir_avg_nuclei=dir_avg_nuclei,
        diameter=config["cellpose"]["diameter"],
        model_type=config["cellpose"]["model"],
        flow_threshold=config["cellpose"]["flow_threshold"],
        cellprob_threshold=config["cellpose"]["cellprob_threshold"],
        pretrained_model=config["cellpose"]["pretrained_model"],
        config=config,
    )

    dir_segmented_nuclei = Path(dir_output, "segmented_dapi")

    config = default_config
    config["cellpose"]["segment_channel"] = 0
    config["cellpose"]["segment_cycle"] = 0
    config["cellpose"]["diameter"] = 37.07
    config["cellpose"][
        "pretrained_model"
    ] = "/cluster/scratch/gutgi/4i/Brain_ECM_4i_2_v2/cellpose_model/CP_20240707_nuclei_lr_0_1_weight_decay_0_0001_epochs_300_dapi_v2"

    config["cellpose"]["cellprob_threshold"] = 0.0
    config["cellpose"]["flow_threshold"] = 0.9
    config["cellpose"]["segment_dual_channel"] = False
    run_cellpose(
        sample=sample,
        dir_images=dir_bg_subtracted,
        dir_output=dir_segmented_nuclei,
        dir_avg_nuclei=dir_avg_nuclei,
        diameter=config["cellpose"]["diameter"],
        model_type=config["cellpose"]["model"],
        flow_threshold=config["cellpose"]["flow_threshold"],
        cellprob_threshold=config["cellpose"]["cellprob_threshold"],
        pretrained_model=config["cellpose"]["pretrained_model"],
        config=config,
    )

    membrane_mask = io.imread(Path(dir_segmented_bcat, sample + ".tif"))
    nuclei_mask = io.imread(Path(dir_segmented_nuclei, sample + ".tif"))
    membrane_mask_expaned = skimage.segmentation.expand_labels(membrane_mask, 30)

    cell_nuclei_assignments = {}
    membrane_props = regionprops(membrane_mask, intensity_image=membrane_mask)
    chosen_nuclei = []
    for cell in tqdm(membrane_props):
        cell_bbox = cell.bbox
        once_cell = (
            membrane_mask[cell_bbox[0] : cell_bbox[2], cell_bbox[1] : cell_bbox[3]]
            == cell.label
        )
        nucleis = nuclei_mask[cell_bbox[0] : cell_bbox[2], cell_bbox[1] : cell_bbox[3]]
        best_iou = 0
        best_nucleus = None

        for nucleus in np.unique(nucleis):
            if nucleus != 0:
                nucleus_mask = nucleis == nucleus
                intersection = np.logical_and(once_cell, nucleus_mask).sum()
                union = np.logical_or(once_cell, nucleus_mask).sum()
                iou = intersection / union
                if not nucleus in chosen_nuclei:
                    if iou > iou_threshold:
                        if iou > best_iou:
                            best_iou = iou
                            best_nucleus = nucleus

        if best_nucleus is not None:
            cell_nuclei_assignments[cell.label] = best_nucleus
            chosen_nuclei.append(best_nucleus)

    # Only keep membrane masks, that are assigned
    membrane_mask_assigned = np.zeros(membrane_mask.shape)
    for cell in tqdm(cell_nuclei_assignments):
        membrane_mask_assigned[membrane_mask == cell] = cell

    # Only keep membrane masks, that are assigned
    nuclei_mask_assigned = np.zeros(nuclei_mask.shape)
    for cell in tqdm(cell_nuclei_assignments):
        nuclei_mask_assigned[nuclei_mask == cell_nuclei_assignments[cell]] = cell

    # Only keep membrane masks, that are assigned
    membrane_mask_expaned_assigned = np.zeros(nuclei_mask.shape)
    for cell in tqdm(cell_nuclei_assignments):
        membrane_mask_expaned_assigned[membrane_mask_expaned == cell] = cell

    # Create final masks
    cytoplasma_mask_final = membrane_mask_assigned * ~(nuclei_mask_assigned > 0)
    ecm_mask_final = membrane_mask_expaned_assigned * ~(membrane_mask_assigned > 0)

    # Save membrane_mask,ecm_mask,nuceli mask
    dir_segmented_cytoplasma = Path(dir_output, "segmented_cytoplasma")
    dir_segmented_ec_niche = Path(dir_output, "segmented_ecm_niche")
    dir_segmented_nuclei = Path(dir_output, "segmented_cell_nuclei")

    dir_segmented_cytoplasma.mkdir(parents=True, exist_ok=True)
    dir_segmented_ec_niche.mkdir(parents=True, exist_ok=True)
    dir_segmented_nuclei.mkdir(parents=True, exist_ok=True)

    io.imsave(
        Path(dir_segmented_nuclei, sample + ".tif"),
        nuclei_mask_assigned.astype(np.uint16),
    )
    io.imsave(
        Path(dir_segmented_ec_niche, sample + ".tif"), ecm_mask_final.astype(np.uint16)
    )
    io.imsave(
        Path(dir_segmented_cytoplasma, sample + ".tif"),
        cytoplasma_mask_final.astype(np.uint16),
    )

    # run extract_features for all
    segmented_list = [
        dir_segmented_cytoplasma,
        dir_segmented_ec_niche,
        dir_segmented_nuclei,
    ]
    for dir_segmented in segmented_list:

        dir_feature_tables = Path(
            dir_output, dir_segmented.name.replace("segmented", "feature_tables")
        )
        dir_feature_tables.mkdir(parents=True, exist_ok=True)

        dir_anndata = Path(
            dir_output, dir_segmented.name.replace("segmented", "anndata")
        )
        dir_anndata.mkdir(parents=True, exist_ok=True)

        extract_features(
            sample=sample,
            dir_images=dir_bg_subtracted,
            dir_output=dir_feature_tables,
            dir_segmented=dir_segmented,
            df_stains=df_stains,
            config=config,
        )

        convert_to_h5ad(
            sample=sample,
            dir_input=dir_feature_tables,
            dir_output=dir_anndata,
            stat=config["analysis_pipeline"]["stat"],
        )


def run_morpho_analysis(sample, default_config):
    dir_segmented_bcat = Path(dir_output, "segmented_bcat")
    dir_speckle_masks = Path(dir_output, "speckle_masks")
    dir_feature_tables_morphometrics = Path(
        dir_output, "feature_tables_morphometrics_cells"
    )
    dir_speckle_masks = Path(dir_output, "speckle_masks")

    run_extract_morphology_features(
        sample=sample,
        dir_output=dir_feature_tables_morphometrics,
        dir_segmented=dir_segmented_cell,
        dir_speckle_masks=dir_speckle_masks,
        config=config,
        dir_segmented_cells=dir_segmented_bcat,
    )
