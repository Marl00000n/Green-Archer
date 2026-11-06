"""Kleine SDF-Bibliothek (Signed Distance Fields) zum prozeduralen Modellieren.

Alle Laengen in Millimetern. Koordinatensystem der Figur:
  x = rechts (aus Sicht des Betrachters der Vorderansicht), y = nach hinten, z = oben.
  Die Figur blickt in Richtung -y (zum Betrachter).
"""
import numpy as np
import trimesh
from skimage import measure

F32 = np.float32


def v3(*a):
    return np.array(a, float)


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def frame_from_axis(axis, hint=(0, -1, 0)):
    """Rotationsmatrix (Zeilen = lokale Achsen u,v,w) mit w = axis."""
    w = unit(axis)
    h = np.asarray(hint, float)
    if abs(np.dot(unit(h), w)) > 0.95:
        h = np.array([1.0, 0, 0])
    u = unit(np.cross(h, w))
    vv = np.cross(w, u)
    return np.array([u, vv, w])


def frame_uvw(u, v, w):
    return np.array([unit(u), unit(v), unit(w)])


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def _local(X, Y, Z, c, R):
    px, py, pz = X - c[0], Y - c[1], Z - c[2]
    if R is None:
        return px, py, pz
    return (R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz,
            R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz,
            R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz)


# ---------------------------------------------------------------- Primitive
# Jede Funktion liefert (fn, bmin, bmax)

def capsule(a, b, ra, rb=None, ripple=None):
    """Konische Kapsel. ripple=(amp, freq_axial, freq_angular, phase) -> Falten."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rb = ra if rb is None else rb
    ba = b - a
    L2 = float(np.dot(ba, ba))
    L = np.sqrt(L2)
    R = frame_from_axis(ba) if ripple is not None else None

    def fn(X, Y, Z):
        px, py, pz = X - a[0], Y - a[1], Z - a[2]
        t = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / L2, 0, 1)
        cx, cy, cz = px - t * ba[0], py - t * ba[1], pz - t * ba[2]
        d = np.sqrt(cx * cx + cy * cy + cz * cz) - (ra + (rb - ra) * t)
        if ripple is not None:
            amp, fa, ft, ph = ripple
            lx = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz
            ly = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz
            s = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz
            th = np.arctan2(ly, lx)
            env = np.sin(np.pi * np.clip(s / L, 0, 1)) ** 0.7
            d = d + amp * np.sin(fa * s + ft * th + ph) * env
        return d
    m = max(ra, rb) + (abs(ripple[0]) if ripple else 0)
    return fn, np.minimum(a, b) - m, np.maximum(a, b) + m


def ellipsoid(c, r, R=None):
    c = np.asarray(c, float); r = np.asarray(r, float)
    rmin = float(r.min())

    def fn(X, Y, Z):
        lx, ly, lz = _local(X, Y, Z, c, R)
        k0 = np.sqrt((lx / r[0]) ** 2 + (ly / r[1]) ** 2 + (lz / r[2]) ** 2)
        k1 = np.sqrt((lx / r[0] ** 2) ** 2 + (ly / r[1] ** 2) ** 2 + (lz / r[2] ** 2) ** 2)
        k1 = np.maximum(k1, 1e-6)
        d = k0 * (k0 - 1.0) / k1
        return np.where(k0 < 1e-4, -rmin, d)
    m = float(r.max())
    return fn, c - m, c + m


def rbox(c, half, R=None, rad=0.5):
    c = np.asarray(c, float); half = np.asarray(half, float)

    def fn(X, Y, Z):
        lx, ly, lz = _local(X, Y, Z, c, R)
        qx = np.abs(lx) - half[0] + rad
        qy = np.abs(ly) - half[1] + rad
        qz = np.abs(lz) - half[2] + rad
        out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
        ins = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
        return out + ins - rad
    m = float(np.linalg.norm(half))
    return fn, c - m, c + m


def cylinder(a, b, r, rad=0.0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ax = b - a
    hh = np.linalg.norm(ax) / 2
    mid = (a + b) / 2
    R = frame_from_axis(ax)

    def fn(X, Y, Z):
        lx, ly, lz = _local(X, Y, Z, mid, R)
        q1 = np.sqrt(lx * lx + ly * ly) - r + rad
        q2 = np.abs(lz) - hh + rad
        return (np.minimum(np.maximum(q1, q2), 0)
                + np.sqrt(np.maximum(q1, 0) ** 2 + np.maximum(q2, 0) ** 2) - rad)
    m = max(r, hh) * 1.8
    return fn, mid - m, mid + m


def torus(c, axis, Rm, r, R_in=None):
    c = np.asarray(c, float)
    R = frame_from_axis(axis)

    def fn(X, Y, Z):
        lx, ly, lz = _local(X, Y, Z, c, R)
        q = np.sqrt(lx * lx + ly * ly) - Rm
        return np.sqrt(q * q + lz * lz) - r
    m = Rm + r
    return fn, c - m, c + m


def custom(fn, bmin, bmax):
    return fn, np.asarray(bmin, float), np.asarray(bmax, float)


# ------------------------------------------------------------------- Feld
class Field:
    def __init__(self, lo, hi, h):
        self.lo = np.asarray(lo, float)
        self.h = float(h)
        self.n = np.ceil((np.asarray(hi, float) - self.lo) / h).astype(int) + 1
        self.f = np.full(tuple(self.n), 30.0, dtype=F32)

    def axes(self):
        return [self.lo[i] + self.h * np.arange(self.n[i]) for i in range(3)]

    def apply(self, shape, k=0.0, mode='add'):
        fn, bmin, bmax = shape
        pad = max(k, 0.0) + 2 * self.h
        i0 = np.floor((bmin - pad - self.lo) / self.h).astype(int)
        i1 = np.ceil((bmax + pad - self.lo) / self.h).astype(int) + 1
        i0 = np.maximum(i0, 0)
        i1 = np.minimum(i1, self.n)
        if np.any(i1 <= i0):
            return
        xs = (self.lo[0] + self.h * np.arange(i0[0], i1[0])).astype(F32)[:, None, None]
        ys = (self.lo[1] + self.h * np.arange(i0[1], i1[1])).astype(F32)[None, :, None]
        zs = (self.lo[2] + self.h * np.arange(i0[2], i1[2])).astype(F32)[None, None, :]
        d = fn(xs, ys, zs)
        shp = (i1 - i0).astype(int)
        d = np.broadcast_to(d, tuple(shp)).astype(F32)
        sl = (slice(i0[0], i1[0]), slice(i0[1], i1[1]), slice(i0[2], i1[2]))
        cur = self.f[sl]
        if mode == 'add':
            new = smin(cur, d, k)
        elif mode == 'sub':
            new = smax(cur, -d, k)
        elif mode == 'int':
            new = smax(cur, d, k)
        else:
            raise ValueError(mode)
        self.f[sl] = new.astype(F32)

    def add(self, shape, k=0.0):
        self.apply(shape, k, 'add')

    def sub(self, shape, k=0.0):
        self.apply(shape, k, 'sub')

    def intersect(self, shape, k=0.0):
        self.apply(shape, k, 'int')

    def sample(self, pts):
        """Trilineares Sampling des Feldes an Weltpunkten (N,3)."""
        from scipy.ndimage import map_coordinates
        pts = np.atleast_2d(pts)
        idx = ((pts - self.lo) / self.h).T
        return map_coordinates(self.f, idx, order=1, mode='nearest')

    def surface_y(self, x, z, y0, y1, step=0.1):
        """Erste Oberflaeche beim Entlangmarschieren von y0 nach y1."""
        ys = np.arange(y0, y1, step if y1 > y0 else -step)
        pts = np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], 1)
        s = self.sample(pts)
        i = np.argmax(s < 0)
        if s[i] >= 0:
            return None
        return ys[i]

    def mesh(self):
        verts, faces, _, _ = measure.marching_cubes(self.f, 0.0137, spacing=(self.h,) * 3)
        verts = verts + self.lo
        m = trimesh.Trimesh(verts, faces, process=False)
        return m


def clean_mesh(m, min_volume=2.0):
    """Kleine losgeloeste Inseln entfernen (Marching-Cubes-Mesh ist selbst bereits wasserdicht)."""
    m.merge_vertices()
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        keep = [p for p in parts if abs(p.volume) > min_volume]
        if not keep:
            keep = [max(parts, key=lambda p: len(p.faces))]
        m = trimesh.util.concatenate(keep)
        m.merge_vertices()
    m.fix_normals()
    return m
