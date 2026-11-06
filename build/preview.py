"""Kleiner Software-Rasterizer (numpy + PIL) fuer Vorschaubilder der STL-Teile."""
import numpy as np
from PIL import Image


def _basis(eye_dir):
    e = np.asarray(eye_dir, float)
    e = e / np.linalg.norm(e)
    f = -e
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(f, up)) > 0.98:
        up = np.array([0.0, -1.0, 0.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def render(parts, eye_dir, path=None, s=3.0, pad=12, bg=(26, 26, 30), bounds=None, crop=None):
    """parts: Liste (trimesh, (r,g,b)). Dreiecke werden dicht abgetastet (Z-Buffer)."""
    r, u, f = _basis(eye_dir)
    e = np.asarray(eye_dir, float); e = e / np.linalg.norm(e)
    light = e + 0.55 * u - 0.45 * r; light /= np.linalg.norm(light)
    L2 = -f * 0.2 + 0.7 * u + 0.5 * r; L2 /= np.linalg.norm(L2)
    Vs, Fs, Ns, Cs = [], [], [], []
    off = 0
    for m, c in parts:
        V = np.asarray(m.vertices); F = np.asarray(m.faces)
        if crop is not None:
            cc = np.asarray(crop, float)
            inb = np.all((V >= cc[0::2]) & (V <= cc[1::2]), axis=1)
            F = F[inb[F].all(1)]
        Vs.append(V); Fs.append(F + off); Ns.append(np.asarray(m.vertex_normals))
        Cs.append(np.tile(np.array(c, float), (len(V), 1)))
        off += len(V)
    V = np.concatenate(Vs); F = np.concatenate(Fs); N = np.concatenate(Ns); C = np.concatenate(Cs)
    sx, sy, dz = V @ r, V @ u, V @ f
    used = np.unique(F)
    if bounds is None:
        bounds = (sx[used].min(), sx[used].max(), sy[used].min(), sy[used].max())
    x0, x1, y0, y1 = bounds
    W = int((x1 - x0) * s) + 2 * pad
    H = int((y1 - y0) * s) + 2 * pad
    PX = (sx - x0) * s + pad
    PY = H - 1 - ((sy - y0) * s + pad)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    front = (fn @ e) > 0
    F = F[front]
    ex = np.stack([PX[F[:, 1]] - PX[F[:, 0]], PX[F[:, 2]] - PX[F[:, 1]], PX[F[:, 0]] - PX[F[:, 2]]], 1)
    ey = np.stack([PY[F[:, 1]] - PY[F[:, 0]], PY[F[:, 2]] - PY[F[:, 1]], PY[F[:, 0]] - PY[F[:, 2]]], 1)
    el = np.sqrt(ex ** 2 + ey ** 2).max(1)
    nsub = np.ceil(el * 1.2).astype(int)
    big = nsub > 14
    nsub = np.clip(nsub, 1, 14)
    Fbig = F[big]; Nbig_idx = np.where(big)[0]
    zbuf = np.full((H, W), np.inf)
    img = np.zeros((H, W, 3), np.float32); img[:] = bg
    shadeN = lambda n: (0.26 + 0.78 * np.clip(n @ light, 0, 1) + 0.2 * np.clip(n @ L2, 0, 1))
    for n in np.unique(nsub[~big]):
        sel = F[(nsub == n) & ~big]
        bary = np.array([(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)])
        for b in bary:
            a0, a1, a2 = sel[:, 0], sel[:, 1], sel[:, 2]
            x = PX[a0] * b[0] + PX[a1] * b[1] + PX[a2] * b[2]
            y = PY[a0] * b[0] + PY[a1] * b[1] + PY[a2] * b[2]
            z = dz[a0] * b[0] + dz[a1] * b[1] + dz[a2] * b[2]
            nn = N[a0] * b[0] + N[a1] * b[1] + N[a2] * b[2]
            nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
            cc = C[a0]
            xi = np.clip(np.round(x).astype(int), 0, W - 1)
            yi = np.clip(np.round(y).astype(int), 0, H - 1)
            order = np.argsort(-z)
            xi, yi, z, nn, cc = xi[order], yi[order], z[order], nn[order], cc[order]
            ok = z < zbuf[yi, xi]
            xi, yi, z, nn, cc = xi[ok], yi[ok], z[ok], nn[ok], cc[ok]
            zbuf[yi, xi] = z
            img[yi, xi] = np.clip(cc * shadeN(nn)[:, None], 0, 255)
    shade1 = lambda n: (0.26 + 0.78 * np.clip(n @ light, 0, 1) + 0.2 * np.clip(n @ L2, 0, 1))
    for tri in Fbig:
        xs = PX[tri]; ys = PY[tri]; zs = dz[tri]
        x0_, x1_ = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), W - 1))
        y0_, y1_ = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), H - 1))
        if x1_ < x0_ or y1_ < y0_:
            continue
        gx, gy = np.meshgrid(np.arange(x0_, x1_ + 1), np.arange(y0_, y1_ + 1))
        den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(den) < 1e-9:
            continue
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
        l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
        z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]
        ok = inside & (z < zbuf[gy, gx])
        nn = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
        nn = nn / max(np.linalg.norm(nn), 1e-9)
        zbuf[gy[ok], gx[ok]] = z[ok]
        img[gy[ok], gx[ok]] = np.clip(C[tri[0]] * shade1(nn), 0, 255)
    im = Image.fromarray(img.astype(np.uint8))
    if path:
        im.save(path)
    return im


VIEWS = {
    'front': (0, -1, 0),
    'back': (0, 1, 0),
    'right': (-1, 0, 0),
    'left': (1, 0, 0),
    'top': (0, -0.02, 1),
    'q34': (0.55, -1, 0.35),
    'q34b': (-0.7, -1, 0.3),
}
