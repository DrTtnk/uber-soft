import bpy
from bpy_extras.object_utils import object_data_add
import numpy as np



def tet_mesh_generator(vertices, tets):
    if "tet_mesh" in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['tet_mesh'].select_set(True)
        bpy.ops.object.delete()

    faces = np.concatenate([
        tets[:, [0, 1, 2]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]],
        tets[:, [1, 2, 3]],
    ])

    mesh = bpy.data.meshes.new("Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    object_data_add(bpy.context, mesh, operator=None, name="tet_mesh")

    return mesh


def get_mesh_volume(verts, tris):
    points = verts[tris]
    return abs(np.sum(points[:, 0] * np.cross(points[:, 1], points[:, 2])) / 6.0)
