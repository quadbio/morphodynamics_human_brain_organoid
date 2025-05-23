import motile
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import skimage
from joblib import Parallel, delayed
from lightinsight.utils.utils import pad_to_shape


def add_nodes(t, min_vol, anisotropy, input_movie, detection_name, zarr_level="0"):
    """Create graph nodes from labeled regions at a specific time point.

    This function extracts regions from a labeled image at time t and adds them as nodes
    to a directed graph. Each node represents a detected object/cell with a feature based on volume.

    Parameters
    ----------
    t : int
        Time point index in the movie
    min_vol : float
        Minimum volume threshold used to calculate volume based feature score.
    anisotropy : float
        Z-axis anisotropy factor for correcting z-coordinate scaling.
    input_movie : zarr.Group or similar
        Hierarchical data structure containing labeled images organized by time and label type.
    detection_name : str
        Name of the detection/label set to use from the input movie.
    zarr_level : str, optional
        Resolution level in the zarr pyramid to use (default is "0" for full resolution)
    Returns
    -------
    nx.DiGraph
        Directed graph containing nodes for all regions detected at time t.
        Each node has attributes: time, show (label), feature (volume-based),
        draw_position, z, y, x coordinates
    """

    graph = nx.DiGraph()
    d = input_movie[str(t)]["labels"][detection_name][zarr_level][:]
    regions = skimage.measure.regionprops(d)
    positions = []
    for i, r in enumerate(regions):
        draw_pos = int(d.shape[0] - r.centroid[0])
        if draw_pos in positions:
            draw_pos += 3
        positions.append(draw_pos)
        graph.add_node(
            r.label - 1,
            time=t,
            show=r.label,
            feature=1
            - np.round(
                min_vol / r.area,
                decimals=3,
            ).item(),
            draw_position=draw_pos,
            z=int(r.centroid[0]) * anisotropy,
            y=int(r.centroid[1]),
            x=int(r.centroid[2]),
        )
    return graph


def extract_edges(
    t,
    input_movie,
    anisotropy,
    max_distance,
    detection_name,
    edge_metrics=["iou_label", "distance", "volume"],
    zarr_level="0",
):
    """Extract candidate edges between regions in consecutive time frames.

    This function identifies potential connections between objects/cells between two time points.
    It calculates similarity metrics (volume, iou, max_intersect, distance...) and creates
    edges where objects are within the maximum distance threshold. If multiple edge_metrics
    are supplied, it will calculate an average of these metrics.
    The function only creates edges where the combined metric score is > 0.

    Parameters
    ----------
    t : int
        Starting time point (edges will be created between t and t+1)
    input_movie : zarr.Group or similar
        Hierarchical data structure containing labeled images
    anisotropy : float
        Z-axis anisotropy factor for distance calculations
    max_distance : float
        Maximum allowed distance between centroids for edge creation
    detection_name : str
        Name of the detection/label set to use
    edge_metrics : list of str, optional
        Metrics to calculate for edge features. Options include:
        - "iou_label": Intersection over Union of label masks
        - "iou": IoU of full segmentation masks
        - "max_intersect": Maximum intersection ratio
        - "distance": Normalized distance feature
        - "volume": Volume similarity feature
    zarr_level : str, optional
        Resolution level in the zarr pyramid (default is "0")

    Returns
    -------
    list of lists
        Each inner list contains [source_node_id, target_node_id, combined_score]
        where the combined score is the average of all requested metrics

    """

    d0 = input_movie[str(t)]["labels"][detection_name][zarr_level][:]
    d1 = input_movie[str(t + 1)]["labels"][detection_name][zarr_level][:]
    region_props_1 = skimage.measure.regionprops_table(
        d0, properties=("label", "centroid", "area", "image")
    )
    region_props_1 = pd.DataFrame(region_props_1)
    region_props_1["centroid-0"] = region_props_1["centroid-0"] * anisotropy
    region_props_2 = skimage.measure.regionprops_table(
        d1, properties=("label", "centroid", "area", "image")
    )
    region_props_2 = pd.DataFrame(region_props_2)
    region_props_2["centroid-0"] = region_props_2["centroid-0"] * anisotropy
    centroid_names = region_props_1.columns[
        region_props_1.columns.str.contains("centroid-")
    ]
    dist_matrix = scipy.spatial.distance_matrix(
        np.array(region_props_1[centroid_names]),
        np.array(region_props_2[centroid_names]),
        p=2,
    )
    edges = []
    for i in range(len(region_props_1)):
        for j in range(len(region_props_2)):
            dist = dist_matrix[i, j]
            if dist < max_distance:
                row_0 = region_props_1.iloc[i]
                row_1 = region_props_2.iloc[j]
                distance_feature = 1 - (dist / max_distance)
                _a0 = row_0["area"]
                _a1 = row_1["area"]
                if _a0 > _a1:
                    area_feature = np.round(
                        _a1 / _a0,
                        decimals=3,
                    ).item()
                else:
                    area_feature = np.round(
                        _a0 / _a1,
                        decimals=3,
                    ).item()

                if "iou" in edge_metrics:
                    image_1 = d0 == row_0["label"]
                    image_2 = d1 == row_1["label"]
                    intersection = np.logical_and(image_1, image_2)
                    union = np.logical_or(image_1, image_2)
                    iou_score = np.sum(intersection) / np.sum(union)
                else:
                    iou_score = 0

                if "max_intersect" in edge_metrics:
                    image_1 = d0 == row_0["label"]
                    image_2 = d1 == row_1["label"]
                    intersection = np.logical_and(image_1, image_2)
                    intersection_score_1 = np.sum(intersection) / np.sum(image_1)
                    intersection_score_2 = np.sum(intersection) / np.sum(image_2)
                    max_intersect_score = np.max(
                        [intersection_score_1, intersection_score_2]
                    )
                else:
                    max_intersect_score = 0

                if "iou_label" in edge_metrics:
                    image_1 = row_0["image"]
                    image_2 = row_1["image"]
                    max_size = np.array([image_1.shape, image_2.shape]).max(0)
                    image_1 = pad_to_shape(image_1, max_size)
                    image_2 = pad_to_shape(image_2, max_size)
                    intersection = np.logical_and(image_1, image_2)
                    union = np.logical_or(image_1, image_2)
                    iou_score_label = np.sum(intersection) / np.sum(union)
                else:
                    iou_score_label = 0

                if "distance" in edge_metrics:
                    distance_feature = distance_feature
                else:
                    distance_feature = 0

                if "volume" in edge_metrics:
                    area_feature = area_feature
                else:
                    area_feature = 0

                if (
                    (
                        distance_feature
                        + area_feature
                        + iou_score
                        + iou_score_label
                        + max_intersect_score
                    )
                    / len(edge_metrics)
                ) > 0.0:
                    edges.append(
                        [
                            row_0["label"] - 1,
                            row_1["label"] - 1,
                            (
                                distance_feature
                                + area_feature
                                + iou_score
                                + iou_score_label
                                + max_intersect_score
                            )
                            / len(edge_metrics),
                        ]
                    )
    return edges


def build_graph(
    max_distance,
    input_movie,
    detection_name,
    time_points,
    min_vol=100,
    anisotropy=1,
    n_jobs=4,
    edge_metrics=["iou", "distance", "volume"],
    zarr_level="0",
):
    """Build a candidate graph from a list of detections.

    This function constructs a complete tracking graph by creating nodes
    for all time points and edges between consecutive frames. It uses parallel processing
    to efficiently handle large datasets.

    Parameters
    ----------
    max_distance : float
        Maximum distance between centroids to place a candidate edge
    input_movie : zarr.Group or similar
        Hierarchical data structure containing labeled images for all time points
    detection_name : str
        Name of the detection/label set to use
    time_points : list of int
        List of time point indices to process
    min_vol : float, optional
        Minimum volume threshold for node features (default is 100)
    anisotropy : float, optional
        Z-axis anisotropy factor for distance calculations (default is 1)
    n_jobs : int, optional
        Number of parallel jobs to use (default is 4)
    edge_metrics : list of str, optional
        Metrics to calculate for edge features (default is ["iou", "distance", "volume"])
    zarr_level : str, optional
        Resolution level in the zarr pyramid (default is "0")

    Returns
    -------
    motile.TrackGraph
        A motile TrackGraph containing the candidate graph with nodes and edges.
        The graph uses "time" as the frame attribute for temporal organization.
    """

    print("Building candidate graph")

    all_graphs = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(add_nodes)(
            i, min_vol, anisotropy, input_movie, detection_name, zarr_level
        )
        for i in time_points
    )
    G = nx.DiGraph()
    for graph in all_graphs:
        if len(graph) > 0:
            G.add_nodes_from(graph.nodes(data=True))

    edges = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(extract_edges)(
            i,
            input_movie,
            anisotropy,
            max_distance,
            detection_name,
            edge_metrics,
            zarr_level,
        )
        for i in time_points[:-1]
    )
    n_e = 0
    edges = [edge for edge in edges if len(edge) > 0]
    for edge in np.vstack(edges):
        G.add_edge(
            edge[0],
            edge[1],
            feature=edge[2],
            edge_id=n_e,
            show="?",
        )
        n_e += 1

    G = motile.TrackGraph(G, frame_attribute="time")

    return G
