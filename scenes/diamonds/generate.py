"""Generates the 'diamonds' sample scene: six loose, gem-quality cut diamonds on draped ivory silk.

Outputs (next to this script):
  meshes/round_brilliant.obj  57-facet round brilliant, ideal (Tolkowsky-style) proportions
  meshes/princess.obj         square princess cut with a two-tier crown and chevron pavilion
  meshes/marquise.obj         marquise brilliant (the round brilliant mapped onto a navette outline)
  meshes/silk.ply             draped silk sheet (smooth normals, uv), binary PLY
  textures/silk_weave.png     satin-weave height map for the silk's bump map
  scene.prscene.json, materials.json, rigs/*.json

All cuts are convex, so each stone is built as the convex hull of its facet vertices: the facets
come out exactly planar and outward-facing (dielectrics need consistent winding). Units: the
round brilliant's girdle diameter is 2 (about 6.5 mm for a 1 ct stone, so 1 unit ~ 3.3 mm).
Requires numpy, scipy and Pillow. Deterministic (fixed seed).
"""

import json
import math
import os
import struct

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)


# ------------------------------------------------------------------------------------------------
# Cuts (y is up; the girdle lies in the xz plane around y = 0; the table faces +y)

def round_brilliant_points(table=0.57, crown=34.5, pavilion=40.75, girdle=0.03, star=0.5, lower=0.77, ring=64):
    """Vertices of a 57-facet round brilliant with girdle radius 1.

    Construction: each bezel (kite) facet k lies in the plane y = g/2 + (1 - rho_k) tan(crown), where
    rho_k is the radial coordinate projected onto azimuth 45k deg; each pavilion main in
    y = -g/2 - (1 - rho_k) tan(pavilion). Star points and lower-girdle points sit on the
    intersection of neighbouring planes, so every kite is exactly planar.
    """
    tc, tp = math.tan(math.radians(crown)), math.tan(math.radians(pavilion))
    g = girdle * 2  # girdle thickness as a fraction of the diameter (2)
    c22 = math.cos(math.radians(22.5))
    pts = []
    for k in range(8):
        a = math.radians(45 * k)
        pts.append((table * math.cos(a), g / 2 + (1 - table) * tc, table * math.sin(a)))  # table corners
        rs = table * c22 + star * (1 - table * c22)
        b = a + math.radians(22.5)
        pts.append((rs * math.cos(b), g / 2 + (1 - rs * c22) * tc, rs * math.sin(b)))  # star points
        rl = 1 - lower
        pts.append((rl * math.cos(b), -g / 2 - (1 - rl * c22) * tp, rl * math.sin(b)))  # lower girdle points
    for i in range(ring):  # round girdle (top and bottom edges)
        a = 2 * math.pi * i / ring
        pts.append((math.cos(a), g / 2, math.sin(a)))
        pts.append((math.cos(a), -g / 2, math.sin(a)))
    pts.append((0.0, -g / 2 - tp, 0.0))  # culet (pointed)
    return np.array(pts)


def princess_points(table=0.70, crown1=0.86, a1=38.0, a2=26.0, girdle=0.02,
                    pav=((0.74, 62.0), (0.46, 52.0), (0.18, 43.0)), culet_angle=34.0, bulge=0.07):
    """Vertices of a princess cut with half-width 1 (square girdle outline).

    Two crown tiers (steeper near the girdle) and a square table. The pavilion is three rings that
    flatten towards the culet (for a convex solid the pavilion profile must bulge below the straight
    girdle-to-culet line, otherwise the rings fall inside the hull); each ring's mid-edge points
    bulge outwards, which folds the pavilion into the chevron facets typical of a princess.
    """
    g = girdle * 2
    y1 = g / 2 + (1 - crown1) * math.tan(math.radians(a1))
    yt = y1 + (crown1 - table) * math.tan(math.radians(a2))
    pts = []
    corners = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    for sx, sz in corners:
        for w, y in ((table, yt), (crown1, y1)):
            pts.append((sx * w, y, sz * w))
    for i in range(4):  # girdle: corners plus points along each edge (straight edges)
        (x0, z0), (x1, z1) = corners[i], corners[(i + 1) % 4]
        for t in np.linspace(0, 1, 5, endpoint=False):
            x, z = x0 + (x1 - x0) * t, z0 + (z1 - z0) * t
            pts.append((x, g / 2, z))
            pts.append((x, -g / 2, z))
    y, w_prev = -g / 2, 1.0
    for w, angle in pav:
        y -= (w_prev - w) * math.tan(math.radians(angle))
        w_prev = w
        for sx, sz in corners:
            pts.append((sx * w, y, sz * w))
        for x, z in ((w, 0), (-w, 0), (0, w), (0, -w)):  # chevrons
            pts.append((x * (1 + bulge), y, z * (1 + bulge)))
    y -= w_prev * math.tan(math.radians(culet_angle))
    pts.append((0.0, y, 0.0))  # culet
    return np.array(pts)


def marquise_radius(phi, half_length=2.0, half_width=1.0):
    """Polar radius of a navette (two circular arcs meeting in points on the x axis)."""
    rc = (half_length ** 2 + half_width ** 2) / (2 * half_width)
    c = rc - half_width
    s = abs(math.sin(phi))
    return -c * s + math.sqrt(c * c * s * s - c * c + rc * rc)


def marquise_points():
    """The round brilliant's vertices mapped onto a 2:1 navette outline (heights unchanged)."""
    out = []
    for x, y, z in round_brilliant_points(table=0.55, ring=96):
        r = math.hypot(x, z)
        if r < 1e-9:
            out.append((x, y, z))
            continue
        phi = math.atan2(z, x)
        R = marquise_radius(phi) * r
        out.append((R * math.cos(phi), y, R * math.sin(phi)))
    return np.array(out)


def hull_mesh(points):
    """Convex hull -> (vertices, triangles) with outward winding."""
    hull = ConvexHull(points)
    used = np.unique(hull.simplices)
    remap = {int(v): i for i, v in enumerate(used)}
    verts = points[used]
    tris = []
    centre = points.mean(axis=0)
    for simplex in hull.simplices:
        a, b, c = (points[i] for i in simplex)
        n = np.cross(b - a, c - a)
        tri = [remap[int(i)] for i in simplex]
        if np.dot(n, a - centre) < 0:
            tri[1], tri[2] = tri[2], tri[1]
        tris.append(tri)
    return verts, np.array(tris)


def write_obj(path, verts, tris, comment):
    with open(path, 'w', newline='\n') as f:
        f.write('# %s\n# %d vertices, %d triangles (flat facets: no normals)\n' % (comment, len(verts), len(tris)))
        for v in verts:
            f.write('v %.7f %.7f %.7f\n' % tuple(v))
        for t in tris:
            f.write('f %d %d %d\n' % (t[0] + 1, t[1] + 1, t[2] + 1))


def bounds(verts):
    return [np.round(verts.min(axis=0), 5).tolist(), np.round(verts.max(axis=0), 5).tolist()]


# ------------------------------------------------------------------------------------------------
# Silk: a draped sheet with soft folds, flatter where the stones lie

SILK_HALF = 26.0
SWELLS = [  # broad undulation: amplitude, wavelength, direction (deg), phase
    (0.30, 14.0, 20.0, 0.3),
    (0.18, 8.5, 105.0, 1.7),
]
# Creases: narrow ridges (+) and valleys (-) along lines, as silk forms when it is gathered.
# point on the line (x, z), direction (deg), height, half-width, length scale
CREASES = [
    ((-6.0, -4.0), 62.0, 0.55, 0.55, 14.0),
    ((-7.5, -1.0), 70.0, -0.30, 0.45, 12.0),
    ((6.5, -5.5), 118.0, 0.60, 0.60, 15.0),
    ((8.0, -2.0), 112.0, -0.35, 0.50, 12.0),
    ((-4.0, 7.5), 150.0, 0.35, 0.45, 10.0),
    ((5.5, 7.0), 30.0, 0.30, 0.40, 9.0),
    ((0.5, -8.5), 92.0, 0.70, 0.75, 16.0),
    ((-1.5, -9.5), 85.0, -0.35, 0.55, 12.0),
    ((-11.0, 5.0), 160.0, 0.50, 0.60, 12.0),
    ((11.5, 4.0), 20.0, 0.45, 0.55, 12.0),
]


def silk_height(x, z):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    r2 = x * x + z * z
    calm = 0.25 + 0.75 * (1 - np.exp(-r2 / 22.0))  # gentler cloth in the middle, where the stones lie
    h = np.zeros_like(x)
    for amp, wl, deg, ph in SWELLS:
        d = math.radians(deg)
        h += amp * np.sin(2 * math.pi * (x * math.cos(d) + z * math.sin(d)) / wl + ph)
    for (px, pz), deg, amp, width, length in CREASES:
        d = math.radians(deg)
        tx, tz = math.cos(d), math.sin(d)
        along = (x - px) * tx + (z - pz) * tz
        across = -(x - px) * tz + (z - pz) * tx
        # Sharp-crested profile (a crease is narrower at the top than a Gaussian), fading at the ends.
        profile = 1 / np.cosh(across / width) ** 2
        h += amp * profile * np.exp(-(along / length) ** 2)
    sag = -0.006 * r2  # the sheet falls away towards its edges
    return h * calm + sag


def silk_normal(x, z, e=1e-3):
    dx = (silk_height(x + e, z) - silk_height(x - e, z)) / (2 * e)
    dz = (silk_height(x, z + e) - silk_height(x, z - e)) / (2 * e)
    n = np.stack([-dx, np.ones_like(dx), -dz], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def write_silk(path, n=300):
    xs = np.linspace(-SILK_HALF, SILK_HALF, n)
    X, Z = np.meshgrid(xs, xs, indexing='xy')
    Y = silk_height(X, Z)
    N = silk_normal(X, Z)
    U = (X + SILK_HALF) / (2 * SILK_HALF)
    V = (Z + SILK_HALF) / (2 * SILK_HALF)
    verts = np.stack([X, Y, Z, N[..., 0], N[..., 1], N[..., 2], U, V], axis=-1).reshape(-1, 8).astype('<f4')
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            faces.append((a, c, b))  # counter-clockwise seen from above (+y)
            faces.append((b, c, d))
    with open(path, 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\ncomment draped ivory silk (generate.py)\n'
                 'element vertex %d\nproperty float x\nproperty float y\nproperty float z\n'
                 'property float nx\nproperty float ny\nproperty float nz\nproperty float u\nproperty float v\n'
                 'element face %d\nproperty list uchar int vertex_indices\nend_header\n' % (len(verts), len(faces))).encode())
        f.write(verts.tobytes())
        packed = bytearray()
        for fa in faces:
            packed += struct.pack('<Biii', 3, *fa)
        f.write(bytes(packed))
    ymin, ymax = float(Y.min()), float(Y.max())
    return [[-SILK_HALF, round(ymin, 4), -SILK_HALF], [SILK_HALF, round(ymax, 4), SILK_HALF]], len(faces)


def write_weave(path, size=512, threads=32):
    """Satin weave height map: long warp floats (along v) crossed by the weft every fifth thread."""
    v, u = np.mgrid[0:size, 0:size] / size * threads
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - iu, v - iv
    warp = 0.5 + 0.5 * np.sin(math.pi * fu) ** 0.6  # rounded warp threads across u
    stitch = (iv * 2 + iu) % 5 == 0  # 5-harness satin: where the weft passes over the warp
    weft = np.sin(math.pi * fv) ** 0.6
    h = np.where(stitch, 0.35 + 0.45 * weft, warp)
    h += 0.04 * rng.standard_normal(h.shape)  # slub: slight irregularity of real silk
    h = (h - h.min()) / (h.max() - h.min())
    Image.fromarray((h * 255 + 0.5).astype(np.uint8), mode='L').save(path)


# ------------------------------------------------------------------------------------------------
# Placement

def rot_euler(ex, ey, ez):
    """Rotation matching the loader's 'euler' (degrees): R = Rz * Ry * Rx."""
    x, y, z = (math.radians(a) for a in (ex, ey, ez))
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def settle(verts, pos_xz, euler, scale, sink):
    """Translation that rests the transformed stone on the silk, its lowest point `sink` into the cloth."""
    R = rot_euler(*euler)
    p = (verts * scale) @ R.T
    lowest = p[np.argmin(p[:, 1])]
    x, z = pos_xz[0] + lowest[0], pos_xz[1] + lowest[2]
    ground = float(silk_height(np.array(x), np.array(z)))
    return [round(pos_xz[0], 4), round(ground - sink - lowest[1], 4), round(pos_xz[1], 4)]


def main():
    os.makedirs(os.path.join(HERE, 'meshes'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'textures'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'rigs'), exist_ok=True)

    cuts = {
        'round_brilliant': (round_brilliant_points(), 'round brilliant, 57 facets, table 57%, crown 34.5 deg, pavilion 40.75 deg'),
        'princess': (princess_points(), 'princess cut, square, two-tier crown, chevron pavilion'),
        'marquise': (marquise_points(), 'marquise brilliant, 2:1 navette outline'),
    }
    geometry = {}
    meshes = {}
    for name, (pts, comment) in cuts.items():
        v, t = hull_mesh(pts)
        write_obj(os.path.join(HERE, 'meshes', name + '.obj'), v, t, comment)
        meshes[name] = v
        geometry[name] = {'type': 'mesh', 'file': 'meshes/%s.obj' % name, 'closed': True, 'bounds': bounds(v)}
        print('%-16s %4d vertices %4d triangles' % (name, len(v), len(t)))
    silk_bounds, nsilk = write_silk(os.path.join(HERE, 'meshes', 'silk.ply'))
    geometry['silk'] = {'type': 'mesh', 'file': 'meshes/silk.ply', 'bounds': silk_bounds}
    print('silk             %d triangles' % nsilk)
    write_weave(os.path.join(HERE, 'textures', 'silk_weave.png'))

    # name, cut, position (x, z), euler (deg), scale, sink into the silk
    stones = [
        ('round 1.0 ct, table up', 'round_brilliant', (0.0, 0.4), (4, 20, -6), 1.00, 0.30),
        ('round 0.5 ct, table down (pavilion up)', 'round_brilliant', (-3.6, 2.1), (172, 5, 8), 0.80, 0.06),
        ('round 0.3 ct, on its side', 'round_brilliant', (3.3, 2.6), (0, 35, 62), 0.66, 0.10),
        ('princess 0.8 ct, table up', 'princess', (-2.6, -2.4), (6, 28, 5), 0.72, 0.30),
        ('princess 0.5 ct, tilted', 'princess', (2.4, -2.1), (-38, -15, 12), 0.60, 0.12),
        ('marquise 0.7 ct, table up', 'marquise', (0.6, 4.2), (-5, -58, 7), 0.62, 0.25),
    ]
    objects = [{'name': 'ivory silk', 'geometry': 'silk', 'material': 'ivory_silk'}]
    for name, cut, xz, euler, scale, sink in stones:
        t = settle(meshes[cut], xz, euler, scale, sink)
        objects.append({'name': name, 'geometry': cut, 'material': 'diamond',
                        'transform': {'translate': t, 'euler': list(euler), 'scale': [scale] * 3}})

    materials = {
        'textures': {
            # ~0.15 mm threads (uv spans the 52-unit sheet)
            'silk_weave': {'type': 'image', 'file': 'textures/silk_weave.png', 'colorspace': 'linear', 'uv_scale': 35}
        },
        'materials': {
            'diamond': {'type': 'dielectric', 'ior': 'diamond'},
            # Satin silk: an ivory base under a rough clear "coat" (the lustre of the smooth floats),
            # plus a sheen lobe for the soft glow at grazing angles.
            'ivory_silk': {
                'type': 'coated_diffuse', 'reflectance': [0.90, 0.84, 0.72],
                'coat_ior': 1.45, 'coat_roughness': 0.28, 'thickness': 0.01,
                'sheen': {'color': [1.0, 0.96, 0.88], 'roughness': 0.3, 'weight': 0.8},
                'bump_map': {'texture': 'silk_weave', 'scale': 0.01},  # subtle: silk threads are ~0.15 mm
            },
        },
    }
    rigs = {
        'jeweller': {'lights': {
            # Small, hard sources are what make diamonds sparkle; the dim softbox and surround keep
            # the silk readable without flooding the stones with light from every direction.
            'key': {'type': 'rect', 'position': [-9, 16, 8], 'look_at': [0, 0, 0], 'size': [1.6, 1.6],
                    'temperature': 5600, 'luminance': 300, 'cos_power': 4},
            'kicker_left': {'type': 'rect', 'position': [-14, 7, -9], 'look_at': [0, 0, 0], 'size': [0.7, 0.7],
                            'temperature': 5200, 'luminance': 700},
            'kicker_right': {'type': 'rect', 'position': [13, 9, -7], 'look_at': [0, 0, 0], 'size': [0.7, 0.7],
                             'temperature': 6000, 'luminance': 700},
            'softbox': {'type': 'rect', 'position': [4, 24, 12], 'look_at': [0, 0, 0], 'size': [14, 9],
                        'temperature': 6000, 'luminance': 0.45},
            'surround': {'type': 'environment', 'color': [0.55, 0.58, 0.66], 'intensity': 0.008},
        }},
        'sparkle': {'lights': dict(
            **{'led_%d' % i: {'type': 'disk', 'radius': 0.35, 'temperature': 5800, 'luminance': 900,
                              'position': [round(15 * math.cos(2 * math.pi * i / 8), 3), 13,
                                           round(15 * math.sin(2 * math.pi * i / 8), 3)],
                              'look_at': [0, 0, 0]} for i in range(8)},
            fill={'type': 'rect', 'position': [0, 26, 6], 'look_at': [0, 0, 0], 'size': [12, 12],
                  'temperature': 6200, 'luminance': 0.8},
            surround={'type': 'environment', 'color': [0.5, 0.52, 0.6], 'intensity': 0.01})},
        'window': {'lights': {
            'window': {'type': 'rect', 'position': [-22, 11, 3], 'look_at': [0, 0, 0], 'size': [8, 12],
                       'temperature': 6500, 'luminance': 12},
            'bounce': {'type': 'environment', 'color': [0.95, 0.9, 0.82], 'intensity': 0.05},
        }},
    }
    for name, rig in rigs.items():
        with open(os.path.join(HERE, 'rigs', name + '.json'), 'w', newline='\n') as f:
            json.dump(rig, f, indent=2)
    with open(os.path.join(HERE, 'materials.json'), 'w', newline='\n') as f:
        f.write('// Materials for the diamonds scene (generated by generate.py).\n')
        json.dump(materials, f, indent=2)

    focus = objects[1]['transform']['translate']
    scene = {
        'version': 1,
        'units': 'meters',
        'include': ['materials.json'],
        'rigs': {name: 'rigs/%s.json' % name for name in rigs},
        'active_rigs': ['jeweller'],
        'geometry': geometry,
        'objects': objects,
        'cameras': {
            'main': {'type': 'thin_lens', 'position': [1.5, 11.0, 12.5], 'look_at': [0.0, 0.0, 0.8], 'fov_y': 30,
                     'f_stop': 8, 'focus_on': focus},
            'closeup': {'type': 'thin_lens', 'position': [1.2, 3.6, 5.2], 'look_at': focus, 'fov_y': 24,
                        'f_stop': 5.6, 'focus_on': focus},
            'top': {'type': 'orthographic', 'position': [0, 20, 0.5], 'look_at': [0, 0, 0.5], 'up': [0, 0, -1],
                    'ortho_height': 12},
        },
        'render': {
            'camera': 'main',
            'seed': 1,
            'integrator': {'type': 'mmlt', 'max_depth': 64, 'mutations_per_pixel': 512,
                           'bootstrap_samples': 4000000, 'chains': 1024},  # large: diamond light is carried by rare, long paths
            'film': {'width': 960, 'height': 540, 'filter': {'type': 'blackman_harris', 'radius': 1.5},
                     'colorspace': 'srgb', 'exposure': 0},
            'output': [{'file': 'render.exr'}, {'file': 'render.png', 'tonemap': 'aces'}],
        },
    }
    with open(os.path.join(HERE, 'scene.prscene.json'), 'w', newline='\n') as f:
        f.write('// Diamonds: three round brilliants, two princess cuts and a marquise, loose on draped ivory\n'
                '// silk. Generated by generate.py (edit that and re-run, rather than this file).\n'
                '// Rigs: jeweller (key + kickers + softbox), sparkle (ring of small LEDs), window.\n')
        json.dump(scene, f, indent=2)
    print('wrote scene.prscene.json')


if __name__ == '__main__':
    main()
