import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymeshfix
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from sklearn import metrics
from trimesh import Trimesh
from trimesh.smoothing import filter_taubin


def extract_mesh(mask):
    vertices, faces, _, _ = marching_cubes(mask, 0, step_size=40)
    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)
    organoid_mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)
    filter_taubin(organoid_mesh, iterations=50)

    surface_normal_starts = organoid_mesh.vertices
    surface_normals = organoid_mesh.vertex_normals
    return organoid_mesh, surface_normal_starts, surface_normals
