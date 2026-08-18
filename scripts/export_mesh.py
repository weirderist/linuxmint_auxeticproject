"""
export_mesh.py — Turns loop centerlines into an actual smooth cylindrical
tube mesh (real vertices + faces, not just point-to-point wireframe
lines), and exports to Wavefront OBJ. Also provides a Blender headless
turntable-render helper.

NOTE ON WHAT'S BEEN TESTED HERE:
The tube-mesh math (generate_tube_mesh, export_to_obj) was written and
run in a sandbox against real numbers -- vertex/face counts and geometry
were checked directly, see the accompanying test file.

The Blender render function (render_turntable_views) could NOT be tested
in that same sandbox -- there's no Blender install available there. It's
written carefully against the standard bpy API, but treat it as
untested until you've run it once yourself. Do a single quick run on one
small OBJ first and check the output images before wiring it into the
bot pipeline, the same way we tested crop_classify.py against a real
image before trusting it blind.
"""

import os
import numpy as np


def _build_frames(centerline: np.ndarray):
    """
    Builds a consistent (tangent, normal, binormal) frame at each point
    along the centerline, using a simple fixed-reference projection
    (robust as long as the path doesn't run near-parallel to the
    reference axis -- true for this project's loop shapes, where z
    displacement is small relative to the in-plane loop size).
    """
    n = centerline.shape[0]
    tangents = np.zeros((n, 3))
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents /= norms

    ref = np.array([0.0, 0.0, 1.0])
    normals = np.zeros((n, 3))
    for i in range(n):
        t = tangents[i]
        if abs(np.dot(t, ref)) > 0.98:
            alt_ref = np.array([1.0, 0.0, 0.0])
            proj = alt_ref - np.dot(alt_ref, t) * t
        else:
            proj = ref - np.dot(ref, t) * t
        pn = np.linalg.norm(proj)
        normals[i] = proj / pn if pn > 1e-9 else np.array([1.0, 0.0, 0.0])

    binormals = np.cross(tangents, normals)
    return tangents, normals, binormals


def generate_tube_mesh(centerline: np.ndarray, radius: float, n_radial: int = 8):
    """
    Sweeps a circular cross-section of the given radius along the
    centerline. Returns (vertices, faces) -- faces are 0-indexed quads
    (v0, v1, v2, v3), caller/exporter converts to OBJ's 1-indexed format.
    The tube is open-ended (not capped, not closed loop-to-loop) -- a
    stated simplification, not an attempt at anatomically closed loops.
    """
    n = centerline.shape[0]
    _, normals, binormals = _build_frames(centerline)

    vertices = np.zeros((n * n_radial, 3))
    angles = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)
    for i in range(n):
        center = centerline[i]
        for j, ang in enumerate(angles):
            offset = radius * (np.cos(ang) * normals[i] + np.sin(ang) * binormals[i])
            vertices[i * n_radial + j] = center + offset

    faces = []
    for i in range(n - 1):
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = i * n_radial + j
            v1 = i * n_radial + j_next
            v2 = (i + 1) * n_radial + j_next
            v3 = (i + 1) * n_radial + j
            faces.append((v0, v1, v2, v3))

    return vertices, faces


def export_loops_to_obj(loops, radius: float, filepath: str, n_radial: int = 8):
    """
    loops: nested list [row][col] -> (n_samples, 3) centerline array
    (flat or relaxed -- caller decides which). Builds a tube for every
    loop and writes one combined OBJ file.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for row in loops:
        for centerline in row:
            verts, faces = generate_tube_mesh(centerline, radius, n_radial=n_radial)
            all_vertices.append(verts)
            for f in faces:
                all_faces.append(tuple(idx + vertex_offset for idx in f))
            vertex_offset += verts.shape[0]

    vertices = np.vstack(all_vertices)

    with open(filepath, "w") as f:
        f.write("# Auxetic knit fabric -- tube mesh export\n")
        f.write(f"# {vertices.shape[0]} vertices, {len(all_faces)} faces\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in all_faces:
            # OBJ is 1-indexed
            f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")

    print(f"Exported {vertices.shape[0]} vertices, {len(all_faces)} faces to {filepath}")
    return vertices.shape[0], len(all_faces)


def render_turntable_views(obj_path: str, output_dir: str, n_views: int = 4,
                             image_size: int = 800):
    """
    UNTESTED IN SANDBOX (no Blender available) -- smoke-test this
    yourself on one small OBJ before relying on it. Shells out to
    Blender's headless CLI to render n_views evenly-spaced camera angles
    around the object.

    Requires `blender` on PATH. Writes PNGs to output_dir, named
    view_00.png .. view_0{n_views-1}.png.
    """
    os.makedirs(output_dir, exist_ok=True)
    blender_script = os.path.join(output_dir, "_render_script.py")

    # KNOWN VERSION-SENSITIVE POINTS (flagged, not guessed silently):
    # - The OBJ import operator changed name between Blender versions
    #   (older: bpy.ops.import_scene.obj, newer 4.x: bpy.ops.wm.obj_import).
    #   Both are attempted below with try/except rather than assuming one.
    # - The render engine identifier also changed across 4.x releases
    #   (BLENDER_EEVEE vs BLENDER_EEVEE_NEXT). Both are attempted.
    # - Camera aiming uses the standard track-to-quaternion idiom
    #   (direction.to_track_quat('-Z','Y')), which is stable, well-known
    #   Blender scripting practice and the part of this script I'm most
    #   confident is correct as written.
    script_contents = f'''
import bpy, math, os
import mathutils

bpy.ops.wm.read_factory_settings(use_empty=True)

try:
    bpy.ops.wm.obj_import(filepath=r"{obj_path}")
except AttributeError:
    bpy.ops.import_scene.obj(filepath=r"{obj_path}")

for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        mat = bpy.data.materials.new(name="LoopMat")
        mat.diffuse_color = (0.75, 0.75, 0.78, 1.0)
        obj.data.materials.append(mat)

bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))

for engine_name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        bpy.context.scene.render.engine = engine_name
        break
    except TypeError:
        continue

bpy.context.scene.render.resolution_x = {image_size}
bpy.context.scene.render.resolution_y = {image_size}

bpy.ops.object.camera_add(location=(0, -10, 5))
cam = bpy.context.object
bpy.context.scene.camera = cam

target = mathutils.Vector((0.0, 0.0, 0.0))
n_views = {n_views}
render_radius = 10.0
for i in range(n_views):
    angle = 2 * math.pi * i / n_views
    cam.location = mathutils.Vector((
        render_radius * math.sin(angle),
        -render_radius * math.cos(angle),
        5.0,
    ))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.context.scene.render.filepath = os.path.join(r"{output_dir}", f"view_{{i:02d}}.png")
    bpy.ops.render.render(write_still=True)
'''
    with open(blender_script, "w") as f:
        f.write(script_contents)

    cmd = f'blender --background --python "{blender_script}"'
    ret = os.system(cmd)
    if ret != 0:
        print(f"WARNING: blender render command returned exit code {ret}. "
              f"Check that 'blender' is on PATH and the script at "
              f"{blender_script} ran without error.")
    return [os.path.join(output_dir, f"view_{i:02d}.png") for i in range(n_views)]
