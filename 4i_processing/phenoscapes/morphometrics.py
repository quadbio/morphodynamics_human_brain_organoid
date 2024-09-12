import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import ndimage
from skimage.measure import regionprops_table,label
from skimage import io
from skimage.morphology import remove_small_holes
import skimage
import math
from pathlib import Path
from sklearn import metrics
from phenoscapes.utils import get_metadata

def extract_contours(mask):
    contour_normals = []
    contour_starts = []
    scale=10
    for sub_mask in np.unique(mask)[1:]:
        one_mask=(mask==sub_mask).astype(np.uint8)
        contours, _ = cv2.findContours(one_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours=contours[0]
        contour_normal=[]
        contour_start=[]
        for i in range(0,len(contours),scale):
            p1 = contours[i - scale][0]
            p2 = contours[i][0]
            v1 = p2 - p1

            normal = np.array([v1[1], -v1[0]])
            normal=normal.astype(float)
            normal /= np.linalg.norm(normal)
            start = (p1 + p2) / 2

            contour_normal.append(normal)
            contour_start.append(start)
        contour_normal=np.array(contour_normal)
        contour_start=np.array(contour_start)
        contour_normals.append(contour_normal)
        contour_starts.append(contour_start)
    return contour_normals,contour_starts

def extract_angles(mask,cell_masks,df,scale):
    contour_normals,contour_starts=extract_contours(mask)
    cosine_sims=[]  
    for cell in df['label']:
        # find largest contour
        cell_mask=(cell_masks==cell).astype(np.uint8)
        contours = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        
        # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree 
        ellipse = cv2.fitEllipse(big_contour)
        (xc,yc),(d1,d2),angle = ellipse
        #print(xc,yc,d1,d2,angle)
        # draw major axis line in red
        rmajor = max(d1,d2)/2
        rminor = min(d1,d2)/2
        angle_rad=np.radians(angle)-np.pi

        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90
        major_axis = np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
        # Find the nearest point on the first contour to the center of the ellipse
        cosine_sim_contour=[]
        for contour_start,contour_normal in zip(contour_starts,contour_normals):
            KD_tree_surface_normals = cKDTree(contour_start, leafsize=100)
            nearest_neighbor = KD_tree_surface_normals.query([xc,yc], k=1)

            # Calculate the normal vector at the nearest point on the first contour
            normal_vector = contour_normal[nearest_neighbor[1]]
            cosine_sim=abs(metrics.pairwise.cosine_similarity([major_axis],
                                                              [normal_vector])[0][0])

            cosine_sim_contour.append([cosine_sim,nearest_neighbor[0]])
        cosine_sims.append(cosine_sim_contour)
        
    cosine_sims=np.array(cosine_sims)
    distances=cosine_sims[:,:,1]
    min_dist_ind=np.argmin(distances,1)
    min_dist=np.min(distances,1)
    
    return cosine_sims,distances,min_dist,min_dist_ind

def run_extract_morphology_features(sample:str,
                     dir_segmented:str,
                     dir_output:str,
                     dir_speckle_masks:str,
                     config:dict = None):
    #Create output dir and get segmentations
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    cell_masks = io.imread(Path(dir_segmented, sample + '.tif'))

    #Extract cell/nuclei morphology info
    df_cell = pd.DataFrame(regionprops_table(label_image=cell_masks,
                                        intensity_image=cell_masks,
                                        properties=('label','centroid',
                                                    'area','major_axis_length',
                                                   'minor_axis_length',"solidity","extent",
                                                    "feret_diameter_max",
                                                    "convex_area","equivalent_diameter")))
    df_cell = df_cell[df_cell['area']>100]
    
    #Extract tissue mask by using the cell masks + smoothing with a gaussian filter
    if config["speckle_removal"]["run"]:
        speckle_dir_sample=Path(dir_speckle_masks, sample)
        df_images = get_metadata(speckle_dir_sample, custom_regex=config['regex'])
        file = df_images[(df_images['channel_id'] == config['cellpose']['segment_channel']) &
                         (df_images['cycle_id'] == config['cellpose']['segment_cycle'])]['file'].values[0]
        speckle_mask = io.imread(Path(speckle_dir_sample, file))
        #Extract tissue mask by using the cell masks + speckle mask
        tissue_mask=(cell_masks>0).astype(float)
        speckle_mask_inside=ndimage.binary_fill_holes(tissue_mask)*(speckle_mask==0)
        #Add speckle mask to introduce empty holes
        tissue_mask=(tissue_mask+speckle_mask_inside>0)
    else:
        #Extract tissue mask by using the cell masks + speckle mask
        tissue_mask=(cell_masks>0).astype(float)
    
    tissue_mask=(skimage.filters.gaussian(tissue_mask,20)>0.5).astype(bool)
    tissue_mask=remove_small_holes(tissue_mask,9000)
    lumen_masks=ndimage.binary_fill_holes(tissue_mask)*~(tissue_mask.astype(bool))
    
    lumen_masks=label(lumen_masks.astype(np.uint8))
    
    lumen_df=pd.DataFrame(regionprops_table(label_image=lumen_masks,
                                        intensity_image=lumen_masks,
                                        properties=('label','centroid',
                                                    'area','major_axis_length',
                                                   'minor_axis_length',"solidity","extent",
                                                    "feret_diameter_max",
                                                    "convex_area","equivalent_diameter")))
    #extract lumen label if there are lumen
    if lumen_masks.max() > 0:
        cosine_sims,distances,min_dist,min_dist_ind=extract_angles(lumen_masks,cell_masks,df_cell,10)
        cosine_sims=np.array(cosine_sims)
        distances=cosine_sims[:,:,1]
        min_dist_ind=np.argmin(distances,1)
        min_dist=np.min(distances,1)
        for column in lumen_df.columns:
            df_cell[f'{column}_lumen']=np.array(lumen_df[column][min_dist_ind])

        cosine_angles=[]
        for sim,index in zip(cosine_sims[:,:,0],min_dist_ind):
                cosine_angles.append(sim[index])

        df_cell['angle_lumen']=np.array(cosine_angles)
        #df['angle_lumen'][minimum_dist]=np.nan
        df_cell['distance_to_lumen']=np.array(min_dist)
    else:
        lumen_cols=['label','centroid', 'area','major_axis_length',
                    'minor_axis_length',"solidity","extent",
                    "feret_diameter_max",
                    "convex_area","equivalent_diameter"]
        for column in lumen_cols:
            df_cell[f'{column}_lumen']=np.nan

        df_cell['angle_lumen']=np.nan
        #df['angle_lumen'][minimum_dist]=np.nan
        df_cell['distance_to_lumen']=np.nan

    #Fill holes to get the organoid mask
    tissue_mask=ndimage.binary_fill_holes(tissue_mask).astype(np.uint8)
    cosine_sims,distances,min_dist,min_dist_ind=extract_angles(tissue_mask,cell_masks,df_cell,10)
    cosine_sims=np.array(cosine_sims)
    distances=cosine_sims[:,:,1]
    min_dist_ind=np.argmin(distances,1)
    min_dist=np.min(distances,1)
    cosine_angles=[]
    for sim,index in zip(cosine_sims[:,:,0],min_dist_ind):
            cosine_angles.append(sim[index])

    df_cell['angle_surface']=np.array(cosine_angles)
    df_cell['distance_surface']=np.array(min_dist)
    df_cell.to_csv(Path(dir_output, sample + '.csv'), index=False)    