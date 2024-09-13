import pickle

import motile
import networkx as nx
import numpy as np
import scipy

# Pretty tqdm progress bars
import zarr
from absl import app, flags
from joblib import Parallel, delayed
from lightinsight.tracking.create_detections import create_detections
from lightinsight.tracking.create_graph import build_graph
from lightinsight.tracking.utils import solution2graph
from lightinsight.utils.utils import max_frame
from motile.constraints import MaxChildren, MaxParents
from motile.costs import EdgeSelection, NodeSelection
from motile.plot import draw_solution, draw_track_graph
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", "", "Input path of the omezarr file")
flags.DEFINE_string("output_path", "", "Output path of the omezarr file")
flags.DEFINE_string("label", "", "Which label to segment")
flags.DEFINE_string("channel", "", "Which channel to segment")
flags.DEFINE_string("max_distance", "", "Maximum neighborhood distance")


def is_ascending_with_step_one(lst):
    # Check if each element in the list is one more than the previous element
    return all(lst[i] + 1 == lst[i + 1] for i in range(len(lst) - 1))


def main(argv):
    zarr_path = FLAGS.input_path
    channel = FLAGS.channel
    output_path = FLAGS.output_path
    label = FLAGS.label

    zarr_array = zarr.open(zarr_path, mode="r")
    input_movie = zarr_array[channel]
    input_shape = input_movie["0"]["labels"][label]["0"].shape
    size_z = input_shape[0]
    size_x = input_shape[1]
    size_y = input_shape[2]
    time_points_channel = sorted(np.array(list(input_movie.group_keys())).astype(int))

    # Create list of all time points with labels
    time_points_labels = [
        t
        for t in time_points_channel
        if "labels" in list(input_movie[str(t)].group_keys())
    ]

    # Create list of all time points containing the label to be tracked
    time_points = [
        t
        for t in time_points_labels
        if label in list(input_movie[str(t)]["labels"].group_keys())
    ]

    # Assert no time_points are missing
    assert is_ascending_with_step_one(time_points)

    chunk_size = ((size_z // 2) + 1, (size_x // 2) + 1, (size_y // 2) + 1)
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store)
    channel_group = root.require_group(channel)
    detection_name = label + "_detection"

    # Create detections movie for tracking
    # Get Max from each tp
    max_vals = Parallel(n_jobs=8, backend="multiprocessing", verbose=5)(
        delayed(max_frame)(time_point, input_movie=input_movie, label=label)
        for time_point in time_points
    )

    cumsum_max_vals = np.cumsum([0] + max_vals[:-1])
    # Create detections_movie
    time_points_done = [
        t
        for t in time_points_labels
        if detection_name in list(input_movie[str(t)]["labels"].group_keys())
    ]
    time_points_detections = np.setdiff1d(time_points_labels, time_points_done)

    if len(time_points_detections) != 0:
        detections = Parallel(n_jobs=8, backend="multiprocessing", verbose=5)(
            delayed(create_detections)(
                time_point,
                input_movie=input_movie,
                channel_group=channel_group,
                cumsum_max=cumsum_max_vals,
                label=label,
                chunk_size=chunk_size,
            )
            for time_point in time_points_detections
        )
    else:
        print("detections did not run")

    # Create detections_movie
    time_points_done = [
        t
        for t in time_points_labels
        if detection_name in list(input_movie[str(t)]["labels"].group_keys())
    ]
    zarr_level = 0
    candidate_graph = build_graph(
        max_distance=35,
        input_movie=input_movie,
        detection_name=detection_name,
        min_vol=1000,
        anisotropy=2 / (0.347 * 2),
        n_jobs=8,
        time_points=time_points_done,
        # edge_metrics=["iou_label", "distance", "volume"],
        edge_metrics=["distance", "iou_label"],
        zarr_level=zarr_level,
    )
    pickle.dump(
        candidate_graph,
        open(output_path + f"{channel}_{label}_track_candidate_graph.pickle", "wb"),
    )

    # create a motile solver
    solver = motile.Solver(candidate_graph)

    solver.add_costs(NodeSelection(weight=-1.0, attribute="feature"))
    solver.add_costs(EdgeSelection(weight=-1.0, attribute="feature"))

    solver.add_constraints(MaxParents(1))
    solver.add_constraints(MaxChildren(1))
    solution = solver.solve(num_threads=16)
    graph, nx_graph = solution2graph(solver, candidate_graph)

    pickle.dump(nx_graph, open(output_path + f"{channel}_{label}_track.pickle", "wb"))
    pickle.dump(
        graph, open(output_path + f"{channel}_{label}_track_motile_graph.pickle", "wb")
    )


if __name__ == "__main__":
    app.run(main)
