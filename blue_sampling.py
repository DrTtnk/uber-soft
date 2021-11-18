import sys
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree


def mesh_area(triangle_list):
    N = np.cross(triangle_list[:, 1] - triangle_list[:, 0], triangle_list[:, 2] - triangle_list[:, 0], axis=1)
    return .5 * np.sqrt(np.sum(N ** 2, axis=1))


reflection = np.array([[0., -1.], [-1., 0.]])


def pick_points(triangle_list):
    # Compute uniform distribution over [0, 1]x[0, 1] lower triangle
    X = np.random.random((triangle_list.shape[0], 2))
    t = np.sum(X, axis=1) > 1
    X[t] = np.dot(X[t], reflection) + 1.

    # Map the [0, 1]x[0, 1] lower triangle to the actual triangles
    return np.einsum('ijk,ij->ik', triangle_list[:, 1:] - triangle_list[:, 0, None], X) + triangle_list[:, 0]


def uniform_sample_mesh(tris, areas, sample_count):
    return pick_points(tris[np.random.choice(tris.shape[0], size=sample_count, p=(areas / np.sum(areas)))])


def blue_noise_sample_elimination(point_list, mesh_surface_area, sample_count):
    # Parameters
    alpha = 8
    rmax = np.sqrt(mesh_surface_area / ((2 * sample_count) * np.sqrt(3.)))

    # Compute a KD-tree of the input point list
    kdtree = KDTree(point_list)

    # Compute the weight for each sample
    D = np.minimum(squareform(pdist(point_list)), 2 * rmax)
    D = (1. - (D / (2 * rmax))) ** alpha

    W = np.zeros(point_list.shape[0])
    for i in range(point_list.shape[0]):
        W[i] = sum(D[i, j] for j in kdtree.query_ball_point(point_list[i], 2 * rmax) if i != j)

    # Pick the samples we need
    heap = sorted((w, i) for i, w in enumerate(W))

    id_set = set(range(point_list.shape[0]))
    while len(id_set) > sample_count:
        # Pick the sample with the highest weight
        w, i = heap.pop()
        id_set.remove(i)

        neighbor_set = set(kdtree.query_ball_point(point_list[i], 2 * rmax))
        neighbor_set.remove(i)
        heap = [(w - D[i, j], j) if j in neighbor_set else (w, j) for w, j in heap]
        heap.sort()

    # Job done
    return point_list[sorted(id_set)]


def sample(triangle_list, sample_count):
    tri_area = mesh_area(triangle_list)
    point_list = uniform_sample_mesh(triangle_list, tri_area, 4 * sample_count)
    return blue_noise_sample_elimination(point_list, np.sum(tri_area), sample_count)
