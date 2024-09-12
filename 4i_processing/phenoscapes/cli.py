# Created by harmelc at 26.08.23
# Title: enter title here
# Description: enter description here
from phenoscapes.registration import run_elastix, run_checks
from phenoscapes.masking import simple_mask, refined_mask
from phenoscapes.cropping_denoising import run_cropping_and_denoising
from phenoscapes.speckle_removal import run_speckle_removal
from phenoscapes.bg_subtraction import run_bg_subtraction
from phenoscapes.segmentation import run_cellpose, generate_label_overlay
from phenoscapes.feature_extraction import extract_features
from phenoscapes.montage import generate_overview_montage
from phenoscapes.sc import convert_to_h5ad, plot_summary
from phenoscapes.morphometrics import run_extract_morphology_features
from pathlib import Path
import yaml
import pandas as pd
import argparse
import sys
# TODO: write class that can be used in normal python scripts as well and is just imported for the cli
#       - implement logger, and write to log file!
#       - implement redo from step X?
#       - save versions of config files...
#       - implement check if all files are there before running a step?
#       - implement loading cellpose model and parameter sweep options
#       - implement cleaning up of directories after pipeline is finished

# TODO: fixes
#      - fix sigma
#      - overflow in ilastik



def load_default_config() -> dict:
    # load default config
    print('Loading default config.')
    def_config = {'regex': None,
                  'metadata': None,
                  'reference_cycle': 1,
                  'stain_nuclei': 'DAPI',
                  'channel_nuclei': 0,
                  'channel_stains': [1, 2, 3],
                  'stitching': {'run': True, 'channel': '01', 'cycle': 'cycle1', 'mask': True},
                  'simple_masking': {'run': True, 'sigma': 100, 'n_binary_operations':50, 'channels':[0, 1, 2, 3]},
                  'alignment': {'run': True, 'reference_cycles': [1, 6, 11, 16], 'mask': False,
                                'param_maps': {'rigid':None, 'affine':None}},
                  'alignment_check': {'run': True},
                  'refined_masking': {'run': True, 'threshold': 0.1, 'sigma': 100,
                                      'use_only_dapi' : False,'n_binary_operations':50, 'cycle_dapi': 1},
                  'speckle_removal':{'run': False,'percentile':2},
                  'cropping_denoising': {'run': True, 'average_nuclei': False,'cropping': True,'mask_zeros_smo' : False,
                                         'denoising': True, 'smo':True, 'multiprocessing': True, 'n_processes': 8},
                  'bg_subtraction': {'run': True, 'bg_cycles': None, 'cycle_dapi': 1, 'cycles': None},
                  'cellpose': {'run': True, 'model': 'cyto2',
                               'diameter': 20, 'flow_threshold': -3, 'cellprob_threshold': 1, 'pretrained_model': False,
                              'segment_channel': False,'segment_cycle': False},
                  'feature_extraction': {'run': True,'expand_labels':None},
                  'feature_morphology_extraction': {'run': False},
                  'analysis_pipeline': {'run': True, 'label_overlay': True, 'overview_montage': True,
                                        'to_h5ad': True, 'overview_plots': True, 'slice_step':5000,
                                       'stat':'median'},
                  'cleanup': {'run': False}}
    return def_config

def read_config(config_file: str) -> dict:
    # read config file
    print(f'Loading config from: {config_file}')
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


# noinspection PyShadowingNames
def process_sample(sample: str,
                   dir_input: str,
                   dir_output: str,
                   metadata_file: str,
                   config_file: str = None) -> None:
    """
    Run pipeline for a given sample.
    :param config_file:
    :param sample:
    :param dir_input:
    :param dir_output:
    :param metadata_file:
    :return:
    """
    # catch std out and err out into file
    # mkdir logs
    Path(dir_output, 'logs').mkdir(parents=True, exist_ok=True)
    sys.stdout = open(Path(dir_output,'logs', f'{sample}.out'), 'w')
    sys.stderr = open(Path(dir_output,'logs', f'{sample}.err'), 'w')

    print('Processing:', sample)
    print('Input directory:', dir_input)
    print('Output directory:', dir_output)

    if config_file is not None:
        # read config file
        print(f'Loading config from: {config_file}')
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            file.close()
    else:
        # load default config
        print('Loading default config.')
        config = load_default_config()

    # load staining metadata
    if config['metadata'] is not None:
        print('Loading metadata specified in config file.')
        metadata_file = config['metadata']
    else:
        print(f'Loading metadata from: {metadata_file}')
        config['metadata'] = metadata_file
    if Path(metadata_file).is_file():
        df_stains = pd.read_csv(metadata_file)
    else:
        raise ValueError('Metadata file not found.')

    if config_file is None:
        print('Saving generated config to output directory.')
        # save config to output directory
        with open(Path(dir_output, 'config.yaml'), 'w') as file:
            yaml.dump(config, file, sort_keys=False)

    # define directories
    dir_stitched = Path(dir_input, 'stitched')
    dir_masks = Path(dir_output, 'masks')
    dir_aligned = Path(dir_output, 'aligned')
    dir_check_aligned = Path(dir_output, 'check_aligned')
    if config['refined_masking']['run']:
        dir_masks_refined = Path(dir_output, 'masks_refined')
    else:
        dir_masks_refined = dir_masks
    dir_speckle_masks = Path(dir_output, 'speckle_masks')
    dir_denoised = Path(dir_output, 'denoised')
    dir_bg_subtracted = Path(dir_output, 'bg_subtracted')
    dir_segmented = Path(dir_output, 'segmented')
    dir_feature_tables = Path(dir_output, 'feature_tables')
    dir_feature_tables_morphometrics = Path(dir_output, 'feature_tables_morphometrics')
    dir_label_overlay = Path(dir_output, 'label_overlay')
    dir_montages = Path(dir_output, 'montages')
    dir_anndata = Path(dir_output, 'anndata')
    dir_overview_plots = Path(dir_output, 'plots_overview')
    dir_avg_nuclei = Path(dir_output, 'avg_nuclei')

    # simple masking
    if config['simple_masking']['run']:
        print('Running initial masking for reference cycle.')
        simple_mask(sample=sample,
                    cycle=config['reference_cycle'],
                    channels = config['simple_masking']['channels'],
                    sigma = config['simple_masking']['sigma'],
                    n_binary = config['simple_masking']['n_binary_operations'],
                    dir_input=dir_stitched,
                    dir_output=dir_masks,
                    config=config)

    # run registration
    if config['alignment']['run']:
        print('Running alignment.')
        if config['alignment']['mask']:
            run_elastix(sample=sample,
                        dir_input=dir_stitched,
                        dir_output=dir_aligned,
                        config=config,
                        dir_masks=dir_masks)
        else:
            run_elastix(sample=sample,
                        dir_input=dir_stitched,
                        dir_output=dir_aligned,
                        config=config,
                        dir_masks=None)

    # run alignment check
    if config['alignment_check']['run']:
        print('Running alignment check.')
        run_checks(sample=sample,
                   dir_input=dir_aligned,
                   dir_output=dir_check_aligned,
                   df_stains=df_stains,
                   config=config)

    # run refined masking
    if config['refined_masking']['run']:
        print('Running refined masking.')
        refined_mask(sample=sample,
                     dir_input=dir_aligned,
                     dir_output=dir_masks_refined,
                     sigma = config['refined_masking']['sigma'],
                     n_binary = config['refined_masking']['n_binary_operations'],
                     use_only_dapi = config['refined_masking']['use_only_dapi'],
                     df_stains=df_stains,
                     config=config)
        
    # run speckle removal
    if config['speckle_removal']['run']:
        print('Running speckle removal.')
        run_speckle_removal(sample=sample,
                                   dir_input=dir_aligned,
                                   dir_masks=dir_masks_refined,
                                   dir_output_masks=dir_speckle_masks,
                                   config=config)

    # run cropping and denoising
    if config['cropping_denoising']['run']:
        print('Running cropping and denoising.')
        run_cropping_and_denoising(sample=sample,
                                   dir_input=dir_aligned,
                                   dir_masks=dir_masks_refined,
                                   dir_output=dir_denoised,
                                   dir_output_masks=dir_masks_refined,
                                   dir_avg_nuclei=dir_avg_nuclei,
                                   config=config)

    # run bg subtraction
    if config['bg_subtraction']['run']:
        print('Running background subtraction.')
        run_bg_subtraction(sample=sample,
                           dir_input=dir_denoised,
                           dir_masks=dir_masks_refined,
                           dir_speckle_masks=dir_speckle_masks,
                           dir_output=dir_bg_subtracted,
                           df_stains=df_stains,
                           config=config)
    
    # run cellpose
    if config['cellpose']['run']:
        run_cellpose(sample=sample,
                     dir_images=dir_bg_subtracted,
                     dir_output=dir_segmented,
                     dir_avg_nuclei=dir_avg_nuclei,
                     diameter=config['cellpose']['diameter'],
                     model_type=config['cellpose']['model'],
                     flow_threshold=config['cellpose']['flow_threshold'],
                     cellprob_threshold=config['cellpose']['cellprob_threshold'],
                     pretrained_model=config['cellpose']['pretrained_model'],
                     config=config)

    # extract features
    if config['feature_extraction']['run']:
        extract_features(sample=sample,
                         dir_images=dir_bg_subtracted,
                         dir_output=dir_feature_tables,
                         dir_segmented=dir_segmented,
                         df_stains=df_stains,
                         config=config)

    # extract features
    if config['feature_morphology_extraction']['run']:
        run_extract_morphology_features(sample=sample,
                         dir_output=dir_feature_tables_morphometrics,
                         dir_segmented=dir_segmented,
                         dir_speckle_masks=dir_speckle_masks,
                         config=config)

    # analysis pipeline and plotting
    if config['analysis_pipeline']['run']:
        if config['analysis_pipeline']['to_h5ad']:
            print('Converting to h5ad.')
            convert_to_h5ad(sample=sample,
                            dir_input=dir_feature_tables,
                            dir_output=dir_anndata,
                            stat = config['analysis_pipeline']['stat'])
        # generate label overlay
        if config['analysis_pipeline']['label_overlay']:
            print('Generating label overlay.')
            generate_label_overlay(sample=sample,
                                   dir_images=dir_bg_subtracted,
                                   dir_segmented=dir_segmented,
                                   dir_output=dir_label_overlay,
                                   config=config)
        # generate overview montage
        if config['analysis_pipeline']['overview_montage']:
            print('Generating overview montage.')
            generate_overview_montage(sample=sample,
                                      dir_images=dir_bg_subtracted,
                                      dir_masks=dir_masks_refined,
                                      dir_output=dir_montages,
                                      df_stains=df_stains,
                                      slice_step=config['analysis_pipeline']['slice_step'],
                                      config=config)
        # plot summary
        if config['analysis_pipeline']['overview_plots']:
            print('Generating overview plots.')
            plot_summary(sample=sample,
                         dir_input=dir_anndata,
                         dir_output=dir_overview_plots)

    # cleanup
    if config['cleanup']['run']:
        print('Cleaning up.')
        # dir_stitched.rmdir()
        # dir_masks.rmdir()
        # dir_denoised.rmdir()

def main():
    """
    Run pipeline for a given sample.
    :return:
    """

    parser = argparse.ArgumentParser(description='Run pipeline for a given sample.')
    parser.add_argument('--sample', type=str, help='sample to process')
    parser.add_argument('--dir_input', type=str, help='directory with input images')
    parser.add_argument('--dir_output', type=str, help='directory for output images')
    parser.add_argument('--metadata_file', type=str, help='path to metadata file')
    parser.add_argument('--config_file', type=str, help='path to config file', default=None)
    parser.add_argument('--generate_default_config', type=bool, default=False, help='generate config file and exit')

    args = parser.parse_args()

    # check for default config
    if args.generate_default_config:
        config = load_default_config()
        if args.config is not None:
            # save config file
            with open(args.config_file, 'w') as file:
                yaml.dump(config, file, sort_keys=False)
                print(f'Saved default config file to: {args.config_file}')
                exit()

    # check metadata file
    if not Path(args.metadata_file).is_file():
        print('Metadata file not found.')
        exit()

    process_sample(sample=args.sample,
                   dir_input=args.dir_input,
                   dir_output=args.dir_output,
                   metadata_file=args.metadata_file,
                   config_file=args.config_file)


if __name__ == '__main__':
    main()


