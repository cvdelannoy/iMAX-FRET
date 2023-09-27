import argparse, sys, os, re

import numpy as np
import plotly.graph_objs as go
import plotly as py
import bpy
import bmesh
from scipy import stats
from itertools import permutations, chain, combinations
from pathlib import Path
from matplotlib import cm
from sklearn.decomposition import PCA
import tqdm
from multiprocessing import Pool
from superpose3d import Superpose3D
from Bio.PDB import PDBParser

sys.path.append(str(Path(__file__).resolve().parents[1]) + '/source/')

from coordinate_recomposition.fret2coords import fret2coords
from helper_functions import get_FRET_distance, parse_output_path, rot2eul, plot_fp_hist


pdb_parser = PDBParser()

# remove cameras and lights from scene
bpy.data.objects['Camera'].select_set(True); bpy.ops.object.delete()
bpy.data.objects['Light'].select_set(True); bpy.ops.object.delete()

global mp_coords_array
global mp_queue

def create_material(color):
    # Create a new material
    material_name = "PointMaterial"
    material = bpy.data.materials.new(name=material_name)
    material.diffuse_color = color
    return material

def save_points_as_obj(coordinates, output_file, n_clust, connect_points=False):
    # Clear existing objects in the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Create a new mesh object
    mesh = bpy.data.meshes.new("PointsOBJ")
    obj = bpy.data.objects.new("PointsOBJ", mesh)

    # Link the mesh object to the scene
    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    # Create a BMesh object and populate it with vertices
    bm = bmesh.new()
    bm_verts = []
    for coord in coordinates:
        bm_verts.append(bm.verts.new(coord))

    # Convert the BMesh to a mesh object
    bm.to_mesh(mesh)
    bm.free()

    # Create material color map based on point density
    nb_points = len(coordinates)
    dfunc_dict = {x: stats.gaussian_kde(coordinates[np.arange(0 + x, nb_points, n_clust), :].T) for x in range(n_clust)}
    dv_array = np.array([dfunc_dict[vi % n_clust].evaluate(vert.co)[0] for vi, vert in enumerate(obj.data.vertices)])
    for x in range(n_clust):
        cur_idx = np.arange(0 + x, nb_points, n_clust)
        dv_array[cur_idx] -= dv_array[cur_idx].min()
        dv_array[cur_idx] /= dv_array[cur_idx].max()
    color_map = cm.get_cmap('coolwarm')

    # Create a sphere for each point and assign material
    for i, vert in enumerate(obj.data.vertices):
        x, y, z = vert.co

        # Create a small sphere at the point's location
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.30, location=(x, y, z))
        sphere_obj = bpy.context.object
        sphere_obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

        # Calculate density value and corresponding color
        color = color_map(dv_array[i])  # Get RGB values

        # Create material and assign to the sphere
        material = create_material(color)
        sphere_obj.active_material = material

    if connect_points:
        # Create edges to connect all points
        num_points = len(coordinates)
        edges = []
        for i in range(num_points - 1):
            edges.append((i, i + 1))
        mesh.edge_keys.foreach_set("index", np.array(edges).flatten())

    # Export the mesh to a wavefront OBJ file
    bpy.ops.export_scene.obj(filepath=output_file)

    # Remove the temporary objects from the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')

    # Remove the mesh object from the scene
    scene.collection.objects.unlink(obj)
    bpy.data.objects.remove(obj)
    bpy.ops.object.delete()

def save_pi_as_obj(coordinates, output_file):

    # Clear existing objects in the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Create a new mesh object
    mesh = bpy.data.meshes.new("PointsOBJ")
    obj = bpy.data.objects.new("PointsOBJ", mesh)

    # Link the mesh object to the scene
    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    # Create a sphere for each point
    for i, cc in enumerate(coordinates):
        p = PCA().fit(cc.T)
        cc_rot = p.transform(cc.T)
        cc_std = np.std(cc_rot, axis=0)
        x, y, z = np.mean(cc.T, axis=0)
        eul_angles = rot2eul(np.linalg.inv(p.components_))
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(x, y, z), rotation=eul_angles, scale=cc_std * 2)

    # Export the mesh to a wavefront OBJ file
    bpy.ops.export_scene.obj(filepath=output_file)

    # Remove the temporary objects from the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')

    # Remove the mesh object from the scene
    scene.collection.objects.unlink(obj)
    bpy.data.objects.remove(obj)
    bpy.ops.object.delete()

def powerset(iterable):
    "from: https://docs.python.org/3/library/itertools.html"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_ndim_powerset(ndim):
    psi_list = []
    for ps in powerset(range(ndim)):
        psi = np.ones(ndim)
        for x in ps:
            psi[x] = -1.0
        psi_list.append(psi)
    return psi_list


def align_coords_to_btn(btn_coords, coords, check_reorder=True):
    if check_reorder:
        sp_list = []
        for psi in (np.array([1,1,1]), np.array([-1,1,1]), np.array([1,-1,1]), np.array([1,1,-1])):
            cc = coords * psi
            sp_list.extend([(Superpose3D(btn_coords, cc[order, :], allow_rescale=False), order, psi) for order in
                            permutations(np.arange(4), 4)])
        sp_list.sort(key=lambda x: x[0][0])
        (rmsd, rmat, tvec, scl), order, psi = sp_list[0]
        coords_fitted = np.dot(rmat, (coords * psi)[order, :].T).T + tvec
    else:
        rmsd, rmat, tvec, scl = Superpose3D(btn_coords, coords, allow_rescale=False)
        coords_fitted = np.dot(rmat, coords.T).T + tvec
    return coords_fitted

def align_coords(coords, align_iters=3):
    centroids = coords[np.random.randint(0, len(coords))] # Start with random coords as centroids
    for i in range(align_iters):
        coords_aligned = [align_coords_to_btn(centroids, cc, check_reorder=i==0) for cc in coords]  # Align others by mirror, rotation and translation
        coords = np.stack(coords_aligned, axis=0)
        centroids = np.mean(coords, axis=0) # take mean of positions as new centroids
    return coords

def parse_rene_txt(txt_fn, target_nb_values):
    mol_list = []
    with open(txt_fn, 'r') as fh:
        txt_str = fh.read()
    for mol_txt in txt_str.split('\n\n')[1:]:
        fp = parse_mol(mol_txt, -1)
        if fp is None: continue
        if len(fp) == 0: continue
        if abs(fp[0]) > 0.1: continue  # not properly corrected or no D-only
        fp = fp[1:]
        if len(fp) == target_nb_values:
            mol_list.append(fp)
    return mol_list

def parse_mol(mt, donor_only_threshold):
    mt_list = mt.split('\n')
    event_dict = {}
    for x in mt_list:
        x = x.strip()
        if not len(x): continue
        if 'No barcode found' in x: return
        if x.startswith('Barcode'):
            barcode_idx = int(x.split(' ')[1][:-1])
            event_dict[barcode_idx] = {}
        else:
            var_name, value = x.rsplit(' ', 1)
            event_dict[barcode_idx][var_name] = float(value)
    fret_fp = [event_dict[ed]['position'] for ed in event_dict]
    fret_fp = [x for x in fret_fp if x > donor_only_threshold]  # filter out donor-only
    return fret_fp


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def pdb_coord(c):
    c_str_list = []
    for nc in c:
        c_str = f'{nc:.3f}'
        c_str = ' ' * (8 - len(c_str)) + c_str
        c_str_list.append(c_str)
    return ''.join(c_str_list)


def coords2pdb(coords):
    txt = ''
    for ci, coord in enumerate(coords):
        txt += f'HETATM{str(ci).rjust(5)}  CA  FRT A{str(ci).rjust(4)}    {pdb_coord(coord)}  1.00  1.00           C\n'
    return txt


def bootstrap_align_parallel(mp_coords_array):
    np.random.seed()  # Required as parallel processes return same number
    nb_molecules = mp_coords_array.shape[0]
    cur_coords_list = mp_coords_array[np.random.randint(0, nb_molecules, nb_molecules), :, :]
    cur_coords_aligned = align_coords(cur_coords_list)
    mean_quad = np.mean(cur_coords_aligned, axis=0)
    return mean_quad


def make_strep_figure(fp_list, pdb_fn, out_dir, bootstrap_iters, cores):

    # --- plot fingrprints and lengths stacked ---
    plot_fp_hist([[f[n] for n in (0,2,4)] for f in fp_list], (10, 5), f'{out_dir}strep_fp.svg')
    # cmap = mpl.colors.LinearSegmentedColormap.from_list('oranges_custom', ('#fee8c8', '#e34a33'))
    # _ = draw_ruler_stack(dist_list, cmap, f'{out_dir}strep_lens_stack.svg')

    # --- compose 3D-coordinates for each fingerprint ---
    coords_list = []
    for fp in fp_list:
        dists = [get_FRET_distance(x) for x in fp]
        coords = fret2coords(dists, 4)
        if coords is None: continue
        coords_list.append(coords[0])

    # --- bootstrap data to generate CIs ---
    if bootstrap_iters > 0:
        print('Bootstrapping data...')
        with Pool(cores) as pool:
            mp_coords_array = np.array(coords_list)
            coords_list = [r for r in tqdm.tqdm(pool.imap_unordered(bootstrap_align_parallel, [mp_coords_array] * bootstrap_iters), total=bootstrap_iters)]
        print('Done')

    # --- align coords to btn from pdb file ---
    if pdb_fn.endswith('.pdb'):
        strep_pdb = pdb_parser.get_structure('strep', pdb_fn)
        btn_c11_list = [[atm for atm in res if atm.name == 'C11'][0] for res in strep_pdb.get_residues() if
                        res.get_resname() == 'BTN']
        btn_c11_coords = np.vstack([atm.coord for atm in btn_c11_list])
    else:
        btn_c11_coords = np.loadtxt(pdb_fn)
    for ci, cur_coords in enumerate(coords_list):
        cur_coords_aligned = align_coords_to_btn(btn_c11_coords, cur_coords)
        coords_list[ci] = cur_coords_aligned
    # coords_array = np.dstack(coords_list)
    # for ki in range(10):
    #     mean_quad = coords_array.mean(axis=2)
    #     mean_quad_center = mean_quad.mean(axis=0)
    #     coords_kabsch_list = [kabsch(c, mean_quad) for c in coords_array.transpose((2,0,1))]
    #     coords_array = np.dstack([c - mean_quad_center for c in coords_kabsch_list])
    coords_array = np.dstack(coords_list)
    coords = np.mean(coords_array, axis=2)
    coords_std = coords_array.std(axis=2)
    np.savetxt(f'{out_dir}quads_std.txt',coords_std)

    # --- draw quads ---
    fig = go.Figure()
    for ci, c in enumerate(coords_array.transpose((2,0,1))):
        cur_quad = np.vstack((c, c[0]))
        fig.add_trace(
            go.Scatter3d(
                x=cur_quad[:, 0],
                y=cur_quad[:, 1],
                z=cur_quad[:, 2],
                mode='lines',
                line=dict(color='grey', width=1),
                name=f'q{ci}'
            )
        )

    end_quad = np.vstack((coords, coords[0]))
    fig.add_trace(
        go.Scatter3d(
            x=end_quad[:, 0],
            y=end_quad[:, 1],
            z=end_quad[:, 2],
            mode='lines',
            line=dict(color='red', width=2),
            name=f'end_quad'
        )
    )

    # Set the layout properties
    fig.update_layout(
        title="3D Line Plot",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis")
    )
    py.offline.plot(fig, filename=f'{out_dir}quads.html', auto_open=False)



    # --- 3d figure ---
    # 1. save coordinates as separate pdb
    coords_pdb = coords2pdb(coords)
    coords_pdb_fn = f'{out_dir}fret_coords.pdb'
    with open(coords_pdb_fn, 'w') as fh:
        fh.write(coords_pdb)

    # save as obj files for blender processing
    quads_coords = np.vstack(coords_list)
    save_points_as_obj(quads_coords, f'{out_dir}individual_quads.obj', 4)
    save_points_as_obj(coords, f'{out_dir}average_quad.obj', 1, connect_points=False)
    save_pi_as_obj(coords_array, f'{out_dir}average_quad_pi.obj')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Given FRET values and streptavidin pdb, relate reconstructed dye '
                                                 'positions to BTN residues.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--fret', type=float, nargs='+', help='List of FRET values')
    input_group.add_argument('--txt', type=str, help='Rene txt file from which correct fps are parsed')
    parser.add_argument('--pdb', type=str, required=True)
    parser.add_argument('--bootstrap-iters', type=int, default=-1,
                        help='Bootstrap coordinates of molecule N times to determine CIs (default: no bootstrap, uncertainty is PI).')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args()
    out_dir = parse_output_path(args.out_dir)
    if args.txt:
        fp_list = parse_rene_txt(args.txt, 3)
        [fp.extend(fp) for fp in fp_list]
        [fp.sort() for fp in fp_list]
    elif args.fret:
        fp_list = [args.fret]
    make_strep_figure(fp_list, args.pdb, out_dir, args.bootstrap_iters, args.cores)
