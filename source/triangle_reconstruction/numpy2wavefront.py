import bpy
import bmesh
import numpy as np
from io import StringIO

def create_material(color):
    # Create a new material
    material_name = "PointMaterial"
    material = bpy.data.materials.new(name=material_name)
    material.diffuse_color = color
    return material

def create_obj_file(coordinates_dict, out_fn, orb_dict={}):
    # Clear existing objects in the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    scene = bpy.context.scene

    # Loop through the dictionary items
    for obj_name, coordinates in coordinates_dict.items():
        # Create a new mesh object
        mesh = bpy.data.meshes.new(obj_name)
        obj = bpy.data.objects.new(obj_name, mesh)

        # Link the mesh object to the scene
        scene.collection.objects.link(obj)

        # Create a BMesh object and populate it with vertices
        bm = bmesh.new()
        for i in range(len(coordinates) - 1):
            v1 = bm.verts.new(coordinates[i])
            v2 = bm.verts.new(coordinates[i + 1])
            bm.edges.new([v1, v2])

        # Convert the BMesh to a mesh object
        bm.to_mesh(mesh)
        bm.free()

    # Add orbs at with centers at given coordinates
    for oid in orb_dict:
        x, y, z = orb_dict[oid]['coords']

        # Create a small sphere at the point's location
        bpy.ops.mesh.primitive_uv_sphere_add(radius=orb_dict[oid]['size'], location=(x, y, z))
        sphere_obj = bpy.context.object
        sphere_obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

        # Calculate density value and corresponding color
        color = orb_dict[oid]['color']  # Get RGB values

        # Create material and assign to the sphere
        material = create_material(color)
        sphere_obj.active_material = material

    # Export the mesh to a wavefront OBJ file
    bpy.ops.export_scene.obj(filepath=out_fn)

#
# # Example usage
# # Generate some random coordinates for demonstration purposes
# np.random.seed(42)
# coordinates_dict = {
#     "Object1": np.random.rand(10, 3),
#     "Object2": np.random.rand(8, 3),
#     "Object3": np.random.rand(12, 3),
# }
#
# create_obj_file(coordinates_dict)
