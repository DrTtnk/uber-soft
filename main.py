import numpy as np
import trimesh
import time

from trimesh import proximity


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Time elapsed: {self.interval} seconds")


def random_point_on_triangle(triangle):
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return u * triangle.a + v * triangle.b + (1 - u - v) * triangle.c


def get_triangle_area(p0, p1, p2):
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))


def generate_buckets(tris):
    areas = [get_triangle_area(t[0], t[1], t[2]) for t in tris]
    return np.cumsum(areas) / np.sum(areas)


def get_bucket_index(buckets, sample):
    left, right = (0, len(buckets) - 1)
    while left <= right:
        mid = (right + left) // 2
        if buckets[mid] < sample:
            left = mid + 1
        elif buckets[mid] > sample:
            right = mid - 1
        else:
            return mid
    return left


def generate_random_points_on_mesh(triangles, num_samples):
    buckets = generate_buckets(triangles)

    points = []
    for _ in range(num_samples):
        points.append(random_point_on_triangle(triangles[get_bucket_index(np.random.uniform(0, 1), buckets)]))

    return np.array(points)


def low_discrepancy_2d(triangles, num_samples):
    buckets = generate_buckets(triangles)
    return [np.random.uniform(buckets[i], buckets[i+1]) for i in range(len(buckets)-1)]


testBuckets = [0, 1, 2, 3, 4, 5, 6, 7]

bucket_index = get_bucket_index(testBuckets, 0.5)


def bucket_sampling(triangles, num_samples):
    buckets = generate_buckets(triangles)
    return [get_bucket_index(buckets, np.random.uniform(0, 1)) for _ in range(num_samples)]


mesh = trimesh.load_mesh('./data/monkey.obj')
mesh.process()

print(f"Mesh loaded: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

pQuery = proximity.ProximityQuery(mesh)

# Initialize np array of vertices with random values
vertices = np.random.rand(10000, 3)

with Timer():
    res = pQuery.on_surface(vertices)

# print(res)
