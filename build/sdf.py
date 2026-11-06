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


def tube(path, r, r_end=None):
    """Rohr entlang eines Polygonzugs (N,3). r darf Skalar oder Array der Laenge N sein."""
    P = np.asarray(path, float)
    A, B = P[:-1], P[1:]
    AB = B - A
    L2 = np.maximum(np.sum(AB * AB, 1), 1e-12)
    r = np.broadcast_to(np.asarray(r, float), (len(P),)).astype(float)
    if r_end is not None:
        r = np.linspace(float(r[0]), float(r_end), len(P))
    ra, rb = r[:-1], r[1:]

    def fn(X, Y, Z):
        shp = np.broadcast(X, Y, Z).shape
        Xb = np.broadcast_to(X, shp)[..., None]
        Yb = np.broadcast_to(Y, shp)[..., None]
        Zb = np.broadcast_to(Z, shp)[..., None]
        px, py, pz = Xb - A[:, 0], Yb - A[:, 1], Zb - A[:, 2]
        t = np.clip((px * AB[:, 0] + py * AB[:, 1] + pz * AB[:, 2]) / L2, 0, 1)
        cx = px - t * AB[:, 0]
        cy = py - t * AB[:, 1]
        cz = pz - t * AB[:, 2]
        d = np.sqrt(cx * cx + cy * cy + cz * cz) - (ra + (rb - ra) * t)
        return d.min(-1).astype(F32)
    m = float(r.max()) + 0.5
    return fn, P.min(0) - m, P.max(0) + m


def mask_of(shapes, soft=1.2, invert=False):
    """Weiche 0..1-Maske aus SDF-Formen (Vereinigung)."""
    fns = [s[0] for s in shapes]

    def m(X, Y, Z):
        d = None
        for fn in fns:
            v = fn(X, Y, Z)
            d = v if d is None else np.minimum(d, v)
        w = np.clip(0.5 - d / soft, 0, 1)
        return 1 - w if invert else w
    return m


def mask_mul(*masks):
    def m(X, Y, Z):
        out = None
        for f in masks:
            v = f(X, Y, Z)
            out = v if out is None else out * v
        return out
    return m


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

    def surface_pts(self, origins, dirs, t0, t1, step=0.12):
        """Erster Oberflaechentreffer entlang Strahlen. -> (Punkte, Treffer-Maske, Normalen)"""
        origins = np.atleast_2d(np.asarray(origins, float))
        dirs = np.atleast_2d(np.asarray(dirs, float))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        ts = np.arange(t0, t1, step)
        P = origins[:, None, :] + dirs[:, None, :] * ts[None, :, None]
        s = self.sample(P.reshape(-1, 3)).reshape(len(origins), len(ts))
        inside = s < 0
        first = np.argmax(inside, 1)
        ok = inside[np.arange(len(origins)), first]
        hit = P[np.arange(len(origins)), first]
        # Verfeinerung per linearer Interpolation zum Nachbarn davor
        prev = np.maximum(first - 1, 0)
        s0 = s[np.arange(len(origins)), prev]
        s1 = s[np.arange(len(origins)), first]
        w = np.where(np.abs(s0 - s1) > 1e-6, s0 / (s0 - s1), 1.0)[:, None]
        hit = P[np.arange(len(origins)), prev] + (hit - P[np.arange(len(origins)), prev]) * np.clip(w, 0, 1)
        return hit, ok, self.normals(hit)

    def normals(self, pts):
        """Gradient des Feldes (Oberflaechennormale) an Weltpunkten."""
        pts = np.atleast_2d(np.asarray(pts, float))
        e = self.h
        g = []
        for ax in range(3):
            d = np.zeros(3); d[ax] = e
            g.append(self.sample(pts + d) - self.sample(pts - d))
        g = np.stack(g, 1)
        nn = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.maximum(nn, 1e-9)

    def displace(self, amp, lam, mask=None, band=2.0, seed=0, octaves=3, gain=0.5,
                 ridged=False, slab=40):
        """Feine Oberflaechenstruktur: fraktales Rauschen, nur in einem Band um die Oberflaeche.

        amp  Amplitude (mm), lam  groesste Wellenlaenge (mm)
        mask Funktion (X,Y,Z)->Gewicht 0..1 (None = ueberall)
        """
        from scipy.ndimage import map_coordinates, gaussian_filter
        rng = np.random.default_rng(seed)
        NC = 48
        grids = []
        for o in range(octaves):
            g = rng.random((NC, NC, NC)).astype(F32) * 2 - 1
            g = gaussian_filter(g, 0.85, mode='wrap')
            g /= max(np.abs(g).max(), 1e-6)
            grids.append(g)
        xs, ys, zs = self.axes()
        for k0 in range(0, self.n[2], slab):
            k1 = min(k0 + slab, self.n[2])
            blk = self.f[:, :, k0:k1]
            w = np.clip(1.0 - np.abs(blk) / band, 0, 1)
            w = w * w * (3 - 2 * w)
            sel = np.nonzero(w > 1e-3)
            if len(sel[0]) == 0:
                continue
            px = xs[sel[0]].astype(F32)
            py = ys[sel[1]].astype(F32)
            pz = zs[k0 + sel[2]].astype(F32)
            wt = w[sel]
            if mask is not None:
                mv = mask(px, py, pz)
                keep = mv > 1e-3
                if not np.any(keep):
                    continue
                sel = tuple(s[keep] for s in sel)
                px, py, pz, wt = px[keep], py[keep], pz[keep], wt[keep]
                wt = wt * mv[keep]
            tot = np.zeros(len(px), F32)
            A, L = 1.0, float(lam)
            nrm = 0.0
            for g in grids:
                c = np.stack([(px / L) % NC, (py / L) % NC, (pz / L) % NC])
                v = map_coordinates(g, c, order=1, mode='grid-wrap').astype(F32)
                if ridged:
                    v = 1.0 - 2.0 * np.abs(v)
                tot += A * v
                nrm += A
                A *= gain
                L *= 0.48
            tot = tot / nrm * (amp * wt)
            self.f[:, :, k0:k1][sel] = (blk[sel] + tot).astype(F32)

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
