#!/usr/bin/env python3
"""The Green Archer (Columbia-Serial 1940) - prozeduraler 3D-Druck-Generator.

Erzeugt mehrteilige, druckoptimierte STL-Dateien:
  figur, bogen_oben, bogen_unten, pfeil, sockel  (+ zusammengebaut als Vorschau)

Aufruf:  python3 green_archer.py [--h 0.4] [--parts figur,bogen,pfeil,sockel] [--out ../stl]
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trimesh
from sdf import *
from sdf import _local

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)


# ============================================================ Pose / Anker
YAW = math.radians(66.0)                      # Kopfdrehung zur Zielrichtung (+x)
U_H = v3(-math.cos(YAW), -math.sin(YAW), 0)   # rechts vom Kopf
V_H = v3(math.sin(YAW), -math.cos(YAW), 0)    # Blickrichtung
W_H = v3(0, 0, 1)
R_H = np.array([U_H, V_H, W_H])
HEAD_C = v3(0, 1.5, 252.5)
HS = 1.08                                      # Kopfgroesse (heroisch)
HEAD_R = (11.3 * HS, 14.4 * HS, 17.4 * HS)


def hw(u, v, w=0.0):
    """Kopf-Koordinaten (rechts, vorne, hoch ueber Kopfmitte) -> Welt."""
    return HEAD_C + HS * (u * U_H + v * V_H + w * W_H)


def hsurf(phi_deg, elev_deg, off=0.0):
    """Punkt auf der Tuchoberflaeche des Kopfes; phi = Azimut (0 = Blickrichtung, + = rechts), elev = Hoehenwinkel."""
    ph, el = math.radians(phi_deg), math.radians(elev_deg)
    d = np.array([math.sin(ph) * math.cos(el), math.cos(ph) * math.cos(el), math.sin(el)])
    r = np.array(HEAD_R) + 0.9
    k0 = math.sqrt(float(np.sum((d / r) ** 2)))
    p = d / k0
    n = unit(p / r ** 2)
    p = p + n * off
    return HEAD_C + p[0] * U_H + p[1] * V_H + p[2] * W_H


MK_W0, MK_A, MK_B = -6.70, 0.045, 0.020      # Kante der Mundbedeckung: w = W0 + A*v + B*u^2


def mask_edge_w(u, vv):
    return MK_W0 + MK_A * vv + MK_B * u * u


def mask_edge(phi_deg, off=0.0):
    """Punkt auf der Kante der Mundbedeckung (Azimut phi in Kopfkoordinaten)."""
    r = np.array(HEAD_R) + 0.3 + off
    lo_, hi_ = -70.0, 40.0
    for _ in range(40):
        el = (lo_ + hi_) / 2
        ph, e = math.radians(phi_deg), math.radians(el)
        d = np.array([math.sin(ph) * math.cos(e), math.cos(ph) * math.cos(e), math.sin(e)])
        q = d / math.sqrt(float(np.sum((d / r) ** 2)))
        if q[2] - mask_edge_w(q[0], q[1]) > 0:
            hi_ = el
        else:
            lo_ = el
    ph, e = math.radians(phi_deg), math.radians((lo_ + hi_) / 2)
    d = np.array([math.sin(ph) * math.cos(e), math.cos(ph) * math.cos(e), math.sin(e)])
    q = d / math.sqrt(float(np.sum((d / r) ** 2)))
    return HEAD_C + q[0] * U_H + q[1] * V_H + q[2] * W_H


def mask_weight(X, Y, Z):
    """1 unterhalb der Tuchkante am Kopf, 0 darueber und ausserhalb des Kopfes."""
    lx, ly, lz = _local(X, Y, Z, HEAD_C, R_H)
    below = np.clip((mask_edge_w(lx, ly) - lz) / 0.45 + 0.5, 0, 1)
    r = np.array(HEAD_R) + 4.0
    q = np.sqrt((lx / r[0]) ** 2 + (ly / r[1]) ** 2 + (lz / r[2]) ** 2)
    near = np.clip((1.12 - q) / 0.14, 0, 1)
    return below * near


def fpt(u, w, off=0.0, grow=0.3):
    """Punkt auf der Gesichtsvorderseite: u seitlich, w hoch (Kopf-Koordinaten)."""
    r = np.array(HEAD_R) + grow
    k = 1.0 - (u / r[0]) ** 2 - (w / r[2]) ** 2
    vv = r[1] * math.sqrt(max(k, 1e-3))
    p = np.array([u, vv, w])
    n = unit(p / r ** 2)
    p = p + n * off
    return HEAD_C + p[0] * U_H + p[1] * V_H + p[2] * W_H


S_L, S_R = v3(27, 0, 219), v3(-27, 0, 219)           # Schultergelenke
E_L, W_L = v3(72, -9, 231), v3(112, -15, 240)         # Bogenarm: Ellbogen, Handgelenk
F_L = v3(120, -15, 240)                                # Faust = Griffmitte des Bogens
E_R, W_R = v3(-57, 6, 236), v3(-21.5, -13.6, 245.5)    # Zugarm
BOW_Y = -15.0                                          # Ebene des Bogens
ARROW_Y, ARROW_Z = -18.7, 248.2                        # Pfeillage
NOCK = v3(-2.0, -18.7, 248.2)                          # Nockpunkt der Sehne (Anker am Kiefer)
ARROW_TIP = 136.0
SIDE = (1, -1)

# Stiefel / Beine
HIP = {s: v3(14 * s, 0, 135) for s in SIDE}
KNEE = {s: v3(32 * s, -3, 73) for s in SIDE}
ANK = {s: v3(47 * s, 0, 11) for s in SIDE}
FOOT_D = {s: unit(v3(0.21 * s, -0.978, 0)) for s in SIDE}   # Zehen leicht nach aussen


def peg_positions():
    out = []
    for s in SIDE:
        for off in (1.0, 14.5):
            p = ANK[s] + FOOT_D[s] * off
            out.append((p[0], p[1]))
    return out


PEG_R, PEG_LEN = 2.8, 6.0


# ====================================================== Hilfsfunktionen
def sdir(a, b):
    return unit(np.asarray(b, float) - np.asarray(a, float))


def catmull(pts, n=40):
    pts = np.asarray(pts, float)
    P = np.vstack([pts[0] * 2 - pts[1], pts, pts[-1] * 2 - pts[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1])
    return np.array(out)


def loft(zlo, zhi, a, b, cx=None, cy=None, mod=None, shell=None, rnd=0.8, zmod=None,
         amp=0.0, rim=0.0, rimw=2.0):
    """Elliptisch gelofteter Koerper um die z-Achse. a,b,cx,cy: Funktionen von z."""
    cx = cx or (lambda z: 0 * z)
    cy = cy or (lambda z: 0 * z)
    zs = np.linspace(zlo, zhi, 12)
    ext = max(float(np.max(a(zs))), float(np.max(b(zs)))) + amp + 2
    mx = float(np.max(np.abs(cx(zs)))) + ext
    my = float(np.max(np.abs(cy(zs)))) + ext

    def fn(X, Y, Z):
        px = X - cx(Z)
        py = Y - cy(Z)
        A, B = a(Z), b(Z)
        q = np.sqrt((px / A) ** 2 + (py / B) ** 2)
        d = (q - 1.0) * np.minimum(A, B)
        if mod is not None:
            d = d + mod(np.arctan2(py, px), Z)
        if shell is not None:
            d = np.abs(d) - shell * 0.5
        zl = zlo + (zmod(np.arctan2(py, px)) if zmod is not None else 0.0)
        if rim:
            d = d - rim * np.exp(-((Z - zl - rimw * 0.55) / rimw) ** 2)
        dz = np.maximum(zl - Z, Z - zhi)
        return smax(d, dz, rnd)
    return fn, v3(-mx, -my, zlo - amp - 3), v3(mx, my, zhi + amp + 3)


def ring(c, axis, Rm, r, **kw):
    return torus(c, axis, Rm, r)


def head_shell(rad, th, wlo, whi):
    """Schale um den Kopf-Ellipsoid zwischen den Hoehen wlo..whi (Kopf-Koordinate w)."""
    r = np.array([HEAD_R[0] + rad, HEAD_R[1] + rad, HEAD_R[2] + rad])

    def fn(X, Y, Z):
        lx, ly, lz = _local(X, Y, Z, HEAD_C, R_H)
        k0 = np.sqrt((lx / r[0]) ** 2 + (ly / r[1]) ** 2 + (lz / r[2]) ** 2)
        k1 = np.maximum(np.sqrt((lx / r[0] ** 2) ** 2 + (ly / r[1] ** 2) ** 2 + (lz / r[2] ** 2) ** 2), 1e-6)
        d = np.abs(k0 * (k0 - 1) / k1) - th * 0.5
        dz = np.maximum(wlo - lz, lz - whi)
        return np.maximum(d, dz)
    return fn, HEAD_C - 30, HEAD_C + 30


# ======================================================= Falten und Naehte
def fold_mod(z0, z1, specs, lo=0.30, hi=1.0, pw=0.8, valley=0.62, vw=0.70):
    """mod(phi, z) fuer loft(): Stofffalten - runde Ruecken, dazwischen scharfe Kniffe."""
    sp = sorted(specs, key=lambda s: s[0])
    mids = []
    for i in range(len(sp)):
        a, b = sp[i], sp[(i + 1) % len(sp)]
        gap = (b[0] - a[0]) % (2 * math.pi)
        mids.append((a[0] + gap / 2, (a[1] + b[1]) / 2,
                     valley * (a[2] + b[2]) / 2, vw * (a[3] + b[3]) / 2))

    def m(phi, z):
        t = np.clip((z - z0) / (z1 - z0), 0, 1)
        scale = lo + (hi - lo) * (1 - t) ** pw
        out = 0.0
        for (p0, drift, amp, w) in sp:                    # Faltenruecken (nach aussen)
            d = (phi - (p0 + drift * t) + np.pi) % (2 * np.pi) - np.pi
            out = out - amp * np.exp(-(d / w) ** 2)
        for (p0, drift, amp, w) in mids:                  # Kniffe (nach innen)
            d = (phi - (p0 + drift * t) + np.pi) % (2 * np.pi) - np.pi
            out = out + amp * np.exp(-np.abs(d) / w)
        return out * scale
    return m


def fold_specs(n, seed, amp=(0.9, 1.9), width=(0.20, 0.34), drift=0.16, phase=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p0 = phase + 2 * math.pi * (i + rng.uniform(-0.22, 0.22)) / n
        out.append((p0, rng.uniform(-drift, drift), rng.uniform(*amp), rng.uniform(*width)))
    return out


def resample(P, pitch):
    """Polygonzug gleichmaessig neu abtasten."""
    P = np.asarray(P, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if s[-1] < pitch:
        return P[:1]
    u = np.arange(0, s[-1], pitch)
    return np.stack([np.interp(u, s, P[:, i]) for i in range(3)], 1)


def project(fld, pts, inward, reach=16.0):
    """Punkte auf die Figurenoberflaeche fallen lassen (Strahl von aussen nach innen)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    inw = np.atleast_2d(np.asarray(inward, float))
    if len(inw) == 1:
        inw = np.repeat(inw, len(pts), 0)
    elif len(inw) != len(pts):                     # auf die Punktzahl umtasten
        u = np.linspace(0, 1, len(inw))
        q = np.linspace(0, 1, len(pts))
        inw = np.stack([np.interp(q, u, inw[:, i]) for i in range(3)], 1)
    inw = inw / np.maximum(np.linalg.norm(inw, axis=1, keepdims=True), 1e-9)
    hit, ok, nrm = fld.surface_pts(pts - inw * reach * 0.5, inw, 0.0, reach, 0.07)
    return hit[ok], nrm[ok]


def stitches(fld, path, inward, pitch=1.7, length=0.95, r=0.30, out=0.10, sub=False, reach=16.0):
    """Steppnaht: kurze Striche entlang eines Pfades, auf die Oberflaeche projiziert."""
    P = resample(path, pitch * 0.5)
    if len(P) < 2:
        return
    hit, nrm = project(fld, P, inward, reach)
    if len(hit) < 2:
        return
    H = resample(hit, pitch)
    if len(H) < 2:
        return
    H, N = project(fld, H, inward, reach=6.0)
    if len(H) < 2:
        return
    T = np.gradient(H, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    for p, n, tg in zip(H, N, T):
        c = p + n * out
        sh = capsule(c - tg * length * 0.5, c + tg * length * 0.5, r)
        (fld.sub if sub else fld.add)(sh, 0.18)


def seam(fld, path, inward, r=0.26, out=0.0, sub=True, pitch=0.5, reach=16.0):
    """Durchgehende Naht-/Kantenrille auf der Oberflaeche."""
    P = resample(path, pitch)
    hit, nrm = project(fld, P, inward, reach)
    if len(hit) < 2:
        return
    pts = hit + nrm * out
    (fld.sub if sub else fld.add)(tube(resample(pts, pitch), r), r * 0.6)


def studs(fld, pts, inward, r=0.55, out=0.28):
    hit, nrm = project(fld, pts, inward)
    for p, n in zip(hit, nrm):
        fld.add(ellipsoid(p + n * out, (r, r, r)), 0.3)


# ================================================================ HAENDE
def finger(fld, joints, radii, inward, crease=0.26, nail=False, k=0.5, swell=0.11):
    """Finger als glatt verlaufendes Rohr mit Gelenkverdickungen, Beugefalten und Nagel.

    joints  3-4 Punkte (MCP, PIP, DIP, Spitze); inward  Richtung zur Beugeseite je Gelenk.
    """
    J = np.array([np.asarray(p, float) for p in joints])
    P = catmull([tuple(p) for p in J], 16)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = seg / seg[-1]
    jt = [float(t[np.argmin(np.linalg.norm(P - J[i], axis=1))]) for i in range(len(J))]
    jt[0], jt[-1] = 0.0, 1.0
    r = np.interp(t, jt, radii)
    for i in range(1, len(J) - 1):                       # Gelenke etwas dicker
        r = r * (1 + swell * np.exp(-((t - jt[i]) / 0.10) ** 2))
    r = r * (1 - 0.22 * np.clip((t - 0.93) / 0.07, 0, 1) ** 2)
    fld.add(tube(P, r), k)
    for i in range(1, len(J) - 1):                       # Beugefalten
        a, b = unit(J[i] - J[i - 1]), unit(J[i + 1] - J[i])
        ax = np.cross(a, b)
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, inward[i])
        ax = unit(ax)
        ri = float(np.interp(jt[i], t, r))
        c = J[i] + unit(inward[i]) * ri * 0.86
        fld.sub(capsule(c - ax * ri * 1.3, c + ax * ri * 1.3, crease), crease * 0.8)
    if nail:
        d = unit(J[-1] - J[-2])
        up = unit(-np.asarray(inward[-1], float))
        rn = float(r[-1])
        c = J[-1] - d * rn * 0.75 + up * rn * 0.80
        fld.sub(ellipsoid(c, (rn * 0.80, rn * 0.60, rn * 0.34), frame_uvw(d, np.cross(up, d), up)), 0.16)


GX, GY, GZ = 120.0, -15.0, 240.0      # Achse des Bogengriffs
GRIP_R = 3.2                           # Radius des Bogengriffs in der Hand
BORE_R = GRIP_R + 0.25                 # Bohrung in der Faust (Spiel)


def _gp(ang_deg, r, z):
    """Punkt in Zylinderkoordinaten um die Bogengriff-Achse."""
    a = math.radians(ang_deg)
    return v3(GX + r * math.cos(a), GY + r * math.sin(a), z)


def build_left_hand(fld):
    """Linke Hand: umfasst den senkrechten Bogengriff (Knoechel zum Betrachter, Daumen oben)."""
    add, sub = fld.add, fld.sub
    # Mittelhand / Handballen (liegt hinten am Griff, Richtung Unterarm)
    Rp = frame_uvw(v3(0.62, 0.78, 0), v3(-0.78, 0.62, 0), v3(0, 0, 1))
    add(rbox(_gp(133, 4.6, GZ + 0.4), (4.9, 3.2, 7.2), Rp, 2.0), 2.0)
    add(ellipsoid(_gp(150, 5.0, GZ - 2.6), (4.4, 3.6, 5.2), Rp), 2.5)          # Kleinfingerballen
    add(ellipsoid(_gp(118, 5.4, GZ + 4.6), (4.2, 3.2, 4.2), Rp), 2.5)          # Daumenballen oben
    add(capsule(v3(112.5, -15.2, 240.0), _gp(150, 4.4, GZ), 5.9, 5.2), 3.0)    # Handgelenk -> Hand
    # vier Finger, um den Griff gelegt
    ANG = (-55.0, 10.0, 62.0, 100.0)
    RAD = (6.60, 5.35, 4.90, 5.00)
    fz = (245.0, 241.9, 238.8, 235.9)
    scale = (1.0, 1.03, 0.97, 0.86)
    thick = (1.80, 1.84, 1.74, 1.52)
    for k, (z, sc, th) in enumerate(zip(fz, scale, thick)):
        tilt = (-1.1, -0.3, 0.5, 1.5)[k]
        J, IN = [], []
        for i, (ag, rr) in enumerate(zip(ANG, RAD)):
            ag2 = ANG[0] + (ag - ANG[0]) * sc
            p = _gp(ag2, rr, z + tilt * i * 0.45)
            J.append(p)
            IN.append(unit(v3(GX, GY, p[2]) - p))                  # Beugeseite = zur Griffachse
        rr = (th * 1.04, th * 0.95, th * 0.85, th * 0.70)
        finger(fld, J, rr, IN, crease=0.28, nail=(k < 2))
        add(ellipsoid(J[0] - unit(IN[0]) * th * 0.55, (th * 0.95, th * 0.95, th * 0.80)), 1.3)   # Knoechel
    # Zwischenfinger-Rillen
    for z in (243.45, 240.35, 237.35):
        sub(tube([_gp(-62, 7.4, z), _gp(0, 6.5, z), _gp(55, 5.6, z), _gp(95, 5.4, z)], 0.34), 0.25)
    # Daumen: liegt vor den Fingern, Spitze zur Mitte
    TJ = [v3(116.8, -18.2, 241.6), v3(121.2, -21.2, 243.6), v3(124.6, -20.4, 244.6), v3(126.6, -18.3, 244.6)]
    TI = [v3(0.2, 0.95, -0.2), v3(0.05, 1.0, -0.1), v3(-0.45, 0.88, -0.1), v3(-0.75, 0.66, -0.1)]
    finger(fld, TJ, (2.05, 1.72, 1.52, 1.26), TI, crease=0.30, nail=True, swell=0.09)
    add(ellipsoid(v3(116.2, -16.8, 241.4), (3.5, 3.1, 4.0)), 2.4)              # Thenar (Daumenballen)
    # Bohrung fuer den Bogengriff
    sub(cylinder(v3(GX, GY, GZ - 11.0), v3(GX, GY, GZ + 11.0), BORE_R))


def string_pt(z):
    """Punkt der Bogensehne auf Hoehe z (Nockpunkt -> Wurfarmspitze)."""
    tipz = BOW_ZG + (BOW_HALF if z >= NOCK[2] else -BOW_HALF)
    tip = v3(bow_x(BOW_HALF), BOW_Y, tipz)
    t = (z - NOCK[2]) / (tip[2] - NOCK[2]) if abs(tip[2] - NOCK[2]) > 1e-9 else 0.0
    return NOCK + (tip - NOCK) * t


def build_right_hand(fld):
    """Rechte Hand: mediterraner Auszug - Zeigefinger ueber, Mittel-/Ringfinger unter dem Pfeil."""
    add, sub = fld.add, fld.sub
    fa = unit(W_R - E_R)
    up = unit(v3(0, 0, 1) - fa * float(np.dot(v3(0, 0, 1), fa)))
    side = unit(np.cross(fa, up))
    Rh = frame_uvw(fa, side, up)
    # Mittelhand
    add(rbox(v3(-18.0, -15.7, 245.4), (5.0, 3.1, 6.4), Rh, 2.0), 2.2)
    add(ellipsoid(v3(-20.0, -14.5, 241.8), (4.0, 3.2, 4.0), Rh), 2.5)          # Kleinfingerballen
    add(capsule(W_R + fa * -2.0, v3(-19.0, -15.3, 245.2), 5.3, 5.0), 3.0)
    # drei Finger am Seil
    for k, (zf, th) in enumerate(((251.9, 1.72), (245.0, 1.76), (241.2, 1.60))):
        sp = string_pt(zf)
        back = unit(v3(-1.0, -0.30, 0.0))                 # von der Sehne weg nach hinten
        inw = unit(v3(0.30, 1.0, 0.0))                     # Beugeseite (zur Handflaeche)
        pal = unit(inw - back * float(np.dot(inw, back)))  # quer dazu, zur Handflaeche
        rd, rt = th * 0.86, th * 0.72
        dz = (0.9, -0.5, -0.9)[k]
        dipp = sp + back * (rd + 1.05) + v3(0, 0, dz * 0.3)      # Sehne in der Endgelenkfalte
        tipp = sp + back * (rd * 0.55 + 0.95) + pal * (rt + 1.2) + v3(0, 0, dz)
        mcp = v3(-14.6, -15.3 - 0.8 * k, zf + (1.6 if k == 0 else -1.0))
        pip = mcp + (dipp - mcp) * 0.60 - pal * 2.5 + v3(0, 0, 0.5 * dz)
        J = [mcp, pip, dipp, tipp]
        IN = [inw, unit(inw * 0.8 + back * 0.4), unit(pal + back * 0.3), unit(back * 0.8 + pal * 0.5)]
        finger(fld, J, (th * 1.05, th * 0.96, rd, rt), IN, crease=0.26, nail=(k < 2))
        add(ellipsoid(mcp - inw * th * 0.5, (th * 0.95, th * 0.95, th * 0.8)), 1.3)
    # Kleiner Finger eingerollt
    PJ = [v3(-19.6, -15.9, 238.8), v3(-16.2, -17.7, 238.2), v3(-14.4, -16.2, 237.5), v3(-15.8, -14.7, 237.4)]
    finger(fld, PJ, (1.44, 1.32, 1.16, 0.98), [v3(0.2, 0.95, 0.2)] * 4, crease=0.24)
    # Daumen angelegt
    TJ = [v3(-21.2, -12.9, 242.2), v3(-17.2, -12.9, 239.8), v3(-13.8, -14.1, 238.6)]
    finger(fld, TJ, (2.02, 1.66, 1.32), [v3(0.15, 0.25, 0.95)] * 3, crease=0.26, nail=True, swell=0.08)
    add(ellipsoid(v3(-20.4, -13.7, 243.6), (3.0, 2.8, 3.4)), 2.4)              # Daumenballen
    # Rille fuer die Sehne in den Fingerkuppen (Sehne bleibt dadurch rund)
    sub(tube([string_pt(z) for z in (236.0, 242.0, 248.199, NOCK[2], 248.201, 254.0, 259.0)], 1.02), 0.25)
    # Sehnen auf dem Handruecken
    for dz in (-1.6, 0.6, 2.8):
        add(tube([v3(-21.0, -14.3, 244.8 + dz), v3(-17.8, -15.7, 245.4 + dz)], 0.55), 0.7)


# ============================================================== FIGUR
def build_figure(h, quick=False):
    fig = Field(lo=(-73, -34, -11), hi=(134, 35, 282), h=h)
    add, sub = fig.add, fig.sub
    log('Figur: Feld', fig.f.shape, f'{fig.f.size / 1e6:.0f} M Voxel')

    # ------------------------------------------------------------ Beine + Stiefel
    for s in SIDE:
        hip, knee, ank = HIP[s], KNEE[s], ANK[s]
        add(capsule(hip, knee, 13.6, 11.0), 6)
        add(capsule(knee, ank + v3(0, 0, 3), 11.0, 7.2), 5)
        Rth = frame_from_axis(knee - hip)
        add(ellipsoid(hip + (knee - hip) * 0.44 + v3(1.6 * s, -4.2, 0), (8.2, 7.6, 20.0), Rth), 5)  # Quadrizeps
        add(ellipsoid(hip + (knee - hip) * 0.42 + v3(-2.6 * s, 4.0, 0), (7.4, 7.0, 18.0), Rth), 5)  # Beuger
        add(ellipsoid(hip + (knee - hip) * 0.70 + v3(-4.4 * s, 1.0, 0), (5.0, 6.4, 11.0), Rth), 5)  # Adduktor
        add(ellipsoid(knee + sdir(knee, ank) * 11.0 + v3(0, 4.6, 0), (6.6, 5.4, 11.0),
                      frame_from_axis(ank - knee)), 4)                                             # Wade
        add(ellipsoid(knee + v3(0, -4.4, -1.0), (6.2, 4.6, 6.0)), 3)                               # Kniescheibe
        def wrinkle(c, axis, rad, span=72.0, rr=0.42, lift=0.0, phase=0.0):
            axu = unit(axis)
            fwd = unit(v3(0, -1, 0) - axu * float(np.dot(v3(0, -1, 0), axu)))
            sid = unit(np.cross(fwd, axu))
            pts = [c + (fwd * math.cos(math.radians(a_ + phase))
                        + sid * math.sin(math.radians(a_ + phase))) * rad
                   + axu * (lift * math.sin(math.radians(a_)))
                   for a_ in np.linspace(-span, span, 11)]
            sub(tube(pts, rr), rr * 0.85)

        for dz in (-5.0, 2.0):                                                      # Kniefalten
            c = knee + v3(0, -5.4, dz)
            sub(capsule(c - v3(4.8, 0, 0.7), c + v3(4.8, 0, -0.7), 0.46), 0.45)
        for (t, rr, lf, sp, ph) in ((0.60, 0.42, 1.6, 54, -26), (0.80, 0.38, -1.3, 46, 22)):
            wrinkle(hip + (knee - hip) * t, knee - hip, 13.6 + (11.0 - 13.6) * t + 0.1,
                    span=sp, rr=rr, lift=lf, phase=ph * s)                          # Oberschenkel
        for (t, rr, lf, sp, ph) in ((0.09, 0.38, 1.2, 50, 30), (0.19, 0.34, -1.0, 42, -24)):
            wrinkle(knee + (ank - knee) * t, ank - knee, 11.0 + (7.2 - 11.0) * t + 0.1,
                    span=sp, rr=rr, lift=lf, phase=ph * s)                          # Wade
        wrinkle(knee + (ank - knee) * 0.14, ank - knee, 11.0 - 3.8 * 0.14 + 0.1,
                span=40, rr=0.34, phase=180)                                        # Kniekehle

        # ------------------------------------------------- Stiefel
        d = FOOT_D[s]                      # Zehenrichtung
        zax = v3(0, 0, 1)
        u = unit(np.cross(d, zax))         # quer zum Fuss
        Rf = np.array([u, d, zax])
        sz = ank[2] - 11.0                 # Unterkante Sohle
        ST = [(-13.8, 4.6, 14.0), (-10.0, 6.0, 18.5), (-5.0, 6.9, 21.0), (0.0, 7.3, 20.5),
              (5.0, 7.5, 16.5), (10.5, 7.4, 13.0), (16.0, 7.0, 10.6), (21.0, 6.2, 8.8),
              (25.5, 5.0, 7.2), (28.6, 3.2, 6.0)]   # (Abstand, halbe Breite, Hoehe ueber Sohle)
        add(shoe(ank, d, u, sz, ST, rad=2.6), 2.2)                                       # Schuh
        add(shoe(ank, d, u, sz, ST, grow=0.5, hcap=2.2, rad=0.7, prof='box'), 0.7)       # Sohle
        add(rbox(ank + d * -7.0 + zax * (sz + 2.6 - ank[2]), (5.6, 5.8, 1.3), Rf, 0.8), 1.0)  # Absatz
        sub(tube([ank + d * 8.0 + u * 7.6 + zax * (sz + 7.0 - ank[2]),
                  ank + d * 20.0 + u * 5.6 + zax * (sz + 6.0 - ank[2]),
                  ank + d * 26.0 + zax * (sz + 5.4 - ank[2]),
                  ank + d * 20.0 - u * 5.6 + zax * (sz + 6.0 - ank[2]),
                  ank + d * 8.0 - u * 7.6 + zax * (sz + 7.0 - ank[2])], 0.42), 0.3)      # Kappennaht
        hc = ank + d * -1.6 + zax * (sz + 1.5 - ank[2])
        sub(capsule(hc - u * 7.0, hc + u * 7.0, 0.5), 0.3)                            # Absatzkante
        for i in range(6):                                                            # Sohlenprofil
            q = ank + d * (3.0 + 4.4 * i) + zax * (sz + 0.40 - ank[2])
            sub(capsule(q - u * 6.6, q + u * 6.6, 0.38), 0.25)
        # Schaft
        sd = unit(knee - ank)
        fw = unit(d - sd * float(np.dot(d, sd)))        # vorn am Schaft
        top = ank + sd * ((57 - ank[2]) / sd[2])
        add(capsule(ank + v3(0, 0, 3.5), top, 7.3, 8.8), 4)
        add(torus(top - sd * 3.2, sd, 9.3, 2.5), 1.2)                 # Umschlag
        add(torus(top - sd * 7.2, sd, 8.9, 0.9), 0.6)                 # Naht unter dem Umschlag
        lc = [(ank + (top - ank) * t, 7.5 + 1.4 * t) for t in (0.22, 0.34, 0.46, 0.58, 0.70, 0.80)]
        ey = []
        for c, rr in lc:                                              # Oesen
            for sg in (-1, 1):
                aa = math.radians(27)
                nd = fw * math.cos(aa) + u * math.sin(aa) * sg
                p = c + nd * rr
                ey.append(p + nd * 1.05)
                add(torus(p + nd * 0.30, nd, 1.15, 0.46), 0.3)
                sub(capsule(p + nd * 0.45, p - nd * 1.8, 0.48), 0.2)
        for off in (0, 1):                                            # Kreuzschnuerung
            pth = [ey[2 * i + ((i + off) % 2)] for i in range(len(lc))]
            add(tube(catmull([tuple(q) for q in pth], 7), 0.72), 0.3)
        kn = ank + (top - ank) * 0.86
        for sg in (-1, 1):                                            # Schleifenenden
            add(tube([kn + fw * 8.8, kn + fw * 9.2 + u * sg * 3.2 + sd * -2.5,
                      kn + fw * 8.4 + u * sg * 5.8 + sd * -7.5], 0.62, 0.42), 0.3)
        add(tube([ank - fw * 6.4 + v3(0, 0, 3.0), top - fw * 8.2], 0.48), 0.3)   # Naht hinten
        for t in (0.14, 0.17):                                        # Falte ueber dem Knoechel
            c = ank + (top - ank) * t
            sub(torus(c, sd, 7.5 + 1.4 * t, 0.40), 0.35)
    log('Beine fertig')

    # --------------------------------------------------------------- Torso
    zz = [112, 118, 135, 150, 165, 180, 195, 208, 218, 224]
    aa = [23.0, 23.5, 23.0, 20.5, 22.0, 25.5, 27.5, 27.0, 23.0, 17.0]
    bb = [15.0, 15.5, 15.5, 14.2, 15.0, 16.5, 17.5, 16.5, 14.5, 12.0]
    fa, fb = PchipInterpolator(zz, aa), PchipInterpolator(zz, bb)
    TS = fold_specs(14, 97, amp=(0.55, 1.20), width=(0.075, 0.135), drift=0.10)
    tbase = fold_mod(112, 224, TS, lo=1.0, hi=1.0, pw=1.0, valley=0.55, vw=0.75)

    def tmod(phi, z):
        return tbase(phi, z) * np.exp(-((z - 168.0) / 30.0) ** 2)
    add(loft(112, 224, lambda z: fa(z), lambda z: fb(z), cy=lambda z: 1.0 + 0 * z,
             mod=tmod, amp=1.6, rnd=2.0), 4)
    for s in SIDE:                                                  # Brustmuskeln
        add(ellipsoid(v3(11.0 * s, -10.5, 194), (9.5, 5.0, 6.8)), 6)
    add(ellipsoid(v3(0, 3, 214), (17, 9, 7)), 5)                    # Trapez
    add(capsule(S_R, S_L, 11.0, 11.0), 6)                           # Schultern
    log('Torso fertig')

    # ----------------------------------------------- Wams-Rock, Guertel, Falten
    szz = [104, 112, 122, 134, 144, 152]
    sfa = PchipInterpolator(szz, [35.0, 33.4, 31.0, 27.4, 23.8, 22.0])
    sfb = PchipInterpolator(szz, [24.6, 23.6, 22.2, 19.6, 17.0, 15.6])

    def skirt_ab(z):
        zc = np.clip(z, 104, 152)
        return sfa(zc), sfb(zc)

    SK = fold_specs(13, 7, amp=(1.5, 2.9), width=(0.065, 0.115), drift=0.22)
    pleat = fold_mod(104, 152, SK, lo=0.14, hi=1.0, pw=0.95)

    def hemz(phi):
        o = 0.0
        for (p0, dr, amp, w) in SK:
            d = (phi - p0 + np.pi) % (2 * np.pi) - np.pi
            o = o + 2.2 * amp / 2.4 * np.exp(-(d / (w * 1.7)) ** 2)
        return 1.8 - o
    add(loft(104, 152, lambda z: skirt_ab(z)[0], lambda z: skirt_ab(z)[1], cy=lambda z: 1.0 + 0 * z,
             mod=pleat, zmod=hemz, amp=3.6, rnd=0.9, rim=0.55, rimw=2.4), 1.4)
    # Seitenschlitze + Mittelschlitz am Saum
    for x0 in (-1, 1):
        sub(rbox(v3(x0 * 36.5, 1.0, 112.0), (3.2, 6.5, 11.0), None, 0.6), 0.6)
    sub(rbox(v3(0, -24.2, 109.0), (0.9, 3.0, 7.0), None, 0.4), 0.3)
    # Blousing ueber dem Guertel
    BL = fold_specs(14, 41, amp=(0.7, 1.4), width=(0.055, 0.09), drift=0.05)
    add(loft(150, 161, lambda z: 22.9 - 0.28 * (z - 150), lambda z: 16.5 - 0.22 * (z - 150),
             cy=lambda z: 1.0 + 0 * z, mod=fold_mod(150, 161, BL, lo=1.0, hi=0.25, pw=1.0),
             amp=1.4, rnd=2.0), 2.2)
    # Guertel
    add(loft(146.0, 153.6, lambda z: 21.9 + 0 * z, lambda z: 15.6 + 0 * z, cy=lambda z: 1.0 + 0 * z,
             shell=2.4, rnd=0.5), 0.5)
    # Schnalle
    add(rbox(v3(0, -15.7, 149.8), (4.6, 1.7, 4.6), None, 0.5), 0.5)
    sub(rbox(v3(0, -17.0, 149.8), (2.5, 1.2, 2.6), None, 0.3), 0.2)
    add(rbox(v3(0, -16.0, 149.8), (0.6, 1.2, 2.6), None, 0.2), 0.2)         # Dorn
    # Beutel an der rechten Huefte, Dolch-Scheide links
    add(ellipsoid(v3(-23.5, -7.5, 142.5), (5.2, 6.4, 7.2)), 2)
    add(ellipsoid(v3(-23.8, -7.5, 148.5), (5.6, 6.8, 3.2)), 1.5)            # Klappe
    sub(capsule(v3(-28.5, -10.0, 147.5), v3(-28.5, -4.0, 147.5), 0.4), 0.2)
    add(ellipsoid(v3(-25.5, -7.5, 137.5), (1.6, 1.6, 1.6)), 0.5)
    log('Wams fertig')

    # ----------------------------------------------------------------- Arme
    ripple_s = (0.2, 1.1, 2.6, 0.0)
    # Bogenarm (links, +x)
    add(ellipsoid(S_L + v3(1.0, 0, 0.5), (13.0, 12.0, 12.5)), 6)
    add(capsule(S_L, E_L, 10.2, 8.3, ripple=ripple_s), 4)
    add(capsule(E_L, W_L, 8.2, 5.6, ripple=(0.16, 1.3, 2.8, 1.0)), 4)
    # Armschiene (Bracer) am linken Unterarm
    b0, b1 = E_L + (W_L - E_L) * 0.30, E_L + (W_L - E_L) * 0.93
    add(capsule(b0, b1, 7.0, 5.6), 0.8)
    for t in (0.34, 0.50, 0.66, 0.82):
        c = E_L + (W_L - E_L) * t
        add(torus(c, W_L - E_L, 7.0 + (5.6 - 7.0) * (t - 0.3) / 0.63 + 0.25, 0.7), 0.4)
    add(torus(b0, W_L - E_L, 7.1, 1.0), 0.6)
    add(torus(b1, W_L - E_L, 5.7, 1.0), 0.6)
    # kurze Wams-Aermel
    for (S, E, sg) in ((S_L, E_L, 1), (S_R, E_R, -1)):
        ad = unit(E - S)
        end = S + (E - S) * 0.56
        rr0, rr1 = 10.9, 9.35
        add(capsule(S - ad * 2.0, end, rr0, rr1), 2.5)
        add(torus(end, ad, rr1 - 0.2, 0.95), 0.5)                       # Saumwulst
        sub(torus(end - ad * 2.4, ad, rr1 + 0.25, 0.38), 0.3)           # Naht
        sd_ = unit(np.cross(ad, v3(0, 0, 1)))
        nd_ = unit(np.cross(sd_, ad))
        for i in range(8):                                              # Falten im Aermel
            a0 = math.radians(-130 + 37 * i)
            r0 = sd_ * math.cos(a0) + nd_ * math.sin(a0)
            p0 = S + ad * 3.0 + r0 * (rr0 + 0.1)
            p1 = end - ad * 2.0 + unit(r0 + sd_ * 0.22) * (rr1 + 0.1)
            sub(tube([p0, (p0 + p1) / 2 + r0 * 0.5, p1], 0.38), 0.45)
    # Zugarm (rechts, -x)
    add(ellipsoid(S_R + v3(-1.0, 0, 0.5), (13.0, 12.0, 12.5)), 6)
    add(capsule(S_R, E_R, 10.2, 8.3, ripple=ripple_s), 4)
    add(capsule(E_R, W_R, 8.2, 5.4, ripple=(0.16, 1.3, 2.8, 2.0)), 4)
    add(torus(W_R + unit(E_R - W_R) * 1.5, W_R - E_R, 5.5, 1.1), 0.6)       # Manschette
    log('Arme fertig')

    # ---------------------------------------------------------- Linke Hand (Bogenhand)
    build_left_hand(fig)
    # ---------------------------------------------------------- Rechte Hand (Zughand)
    build_right_hand(fig)
    log('Haende fertig')

    # --------------------------------------------------------- Kopf, Tuch, Kappe
    add(ellipsoid(HEAD_C, HEAD_R, R_H), 2)                                        # Schaedel
    add(ellipsoid(hw(0, 1.6, -7.4), (9.6, 11.8, 8.4), R_H), 4)                    # Kiefer
    add(ellipsoid(hw(0, 7.0, -11.0), (4.8, 4.6, 3.6), R_H), 3)                    # Kinn
    for sg in (-1, 1):                                                            # Wangenknochen
        add(ellipsoid(hsurf(sg * 33, 2, -2.2), (4.6, 4.0, 3.2), R_H), 3.0)
        add(ellipsoid(hsurf(sg * 62, -12, -2.0), (3.6, 3.4, 4.6), R_H), 3.0)      # Kiefermuskel
    # Nase
    add(capsule(fpt(0, 6.4, -1.5), fpt(0, -2.4, -1.1), 1.05, 1.45), 1.5)
    add(ellipsoid(fpt(0, -4.4, -1.2), (1.80, 1.7, 1.45), R_H), 1.2)               # Nasenspitze
    for sg in (-1, 1):
        add(ellipsoid(fpt(sg * 1.9, -4.9, -1.2), (1.10, 1.05, 1.0), R_H), 1.0)    # Nasenfluegel
        sub(tube([fpt(sg * 1.6, -5.9, 0.10), fpt(sg * 2.7, -4.1, 0.10)], 0.26), 0.22)
        sub(ellipsoid(fpt(sg * 1.5, -5.6, 0.0), (0.55, 0.5, 0.45), R_H), 0.2)     # Nasenloch
    # Brauenwulst und Stirn
    add(tube([fpt(u, 6.1 - 0.022 * u * u, -0.55) for u in np.arange(-9.5, 9.6, 1.0)], 1.30), 1.1)
    sub(tube([fpt(u, 8.4 - 0.020 * u * u, 0.28) for u in np.arange(-8.0, 8.1, 1.0)], 0.32), 0.3)
    sub(tube([fpt(0, 6.6, 0.25), fpt(0, 3.4, 0.25)], 0.30), 0.3)                  # Zornesfalte
    # Augen: Oeffnung im Tuch, Augapfel, Lider
    for sg in (-1, 1):
        c = fpt(sg * 4.8, 2.1, 0.0)
        vo = unit(c - HEAD_C - W_H * float(np.dot(c - HEAD_C, W_H)) * 0.55)
        uo = unit(np.cross(W_H, vo))
        Re = np.array([uo, vo, W_H])
        # flache Mandelform: breiter Schlitz, Augapfel buendig dahinter
        sub(ellipsoid(c + vo * 0.30, (3.05, 2.3, 1.38), Re), 0.28)                # Oeffnung
        sub(ellipsoid(c + vo * 0.10 - W_H * 0.95, (2.3, 1.6, 0.75), Re), 0.25)    # untere Lidkante
        add(ellipsoid(c - vo * 2.20, (2.20, 2.1, 2.00), Re), 0.30)                # Augapfel, buendig
        sub(ellipsoid(c - vo * 2.55, (0.62, 0.62, 0.62), Re), 0.14)               # Pupille
        add(tube([fpt(sg * (4.8 + k), 3.9 - 0.028 * k * k, -0.15)
                  for k in np.arange(-3.0, 3.1, 0.5)], 0.62), 0.42)               # Oberlid
        add(tube([fpt(sg * (4.8 + k), 0.2 + 0.020 * k * k, -0.45)
                  for k in np.arange(-2.5, 2.6, 0.5)], 0.34), 0.35)               # Unterlid
        sub(tube([fpt(sg * (4.8 + k), 4.9 - 0.028 * k * k, 0.18)
                  for k in np.arange(-3.0, 3.1, 0.5)], 0.24), 0.25)               # Lidfalte
    # Mundpartie: nur als weiche Form, sie liegt unter der Tuchlage
    add(tube([fpt(u, -8.2 + 0.045 * u * u, -1.3) for u in np.arange(-4.0, 4.1, 0.6)], 1.20), 1.8)
    add(tube([fpt(u, -10.1 + 0.050 * u * u, -1.4) for u in np.arange(-3.6, 3.7, 0.6)], 1.10), 1.8)
    add(ellipsoid(fpt(0, -11.6, -1.3), (3.4, 2.8, 2.2), R_H), 1.8)                # Kinnpolster

    # ---------------------------------------- Mundbedeckung (Tuch bis unter die Nase)
    fig.offset_region(0.85, mask_weight)
    MED = [mask_edge(a_, 0.95) for a_ in np.arange(-112, 113, 3.5)]
    add(tube(MED, 0.70), 0.45)                                                    # Umschlagkante
    sub(tube([mask_edge(a_, 1.95) for a_ in np.arange(-110, 111, 3.5)], 0.26), 0.25)
    MST = [mask_edge(a_, 1.25) for a_ in np.arange(-108, 109, 2.0)]
    stitches(fig, np.array(MST), np.stack([HEAD_C - q for q in MST]),
             pitch=1.9, length=0.8, r=0.22, out=0.06)
    for sg in (-1, 1):                                                            # Zugfalten im Tuch
        for (p0, p1, rr) in ((16, 62, 0.30), (34, 86, 0.27)):
            pa = mask_edge(sg * p0, 1.1)
            pb = hsurf(sg * p1, -42, 1.2)
            pc = hsurf(sg * (p1 + 16), -22, 1.2)
            sub(tube(catmull([tuple(pa), tuple(pb), tuple(pc)], 8), rr), rr * 0.9)
    sub(tube([hsurf(0, -58, 1.2), hsurf(0, -76, 1.2)], 0.24), 0.3)
    # weiche Zugfalten des Tuchs (Wange -> Ohr)
    for sg in (-1, 1):
        sub(tube([fpt(sg * 3.0, -6.2, 0.10), fpt(sg * 6.6, -8.6, 0.10),
                  hsurf(sg * 52, -24, 0.10), hsurf(sg * 72, -30, 0.10)], 0.26), 0.35)
        add(tube([hsurf(sg * 86, 6, -0.3), hsurf(sg * 92, -16, -0.3)], 1.2), 1.1)  # Ohr unter dem Tuch
    # Knoten im Nacken + Zipfel
    add(ellipsoid(hsurf(180, -14, 1.6), (3.4, 2.9, 3.0), R_H), 1.2)
    sub(tube([hsurf(176, -14, 2.0), hsurf(184, -14, 2.0)], 0.42), 0.35)
    add(tube([hsurf(178, -18, 1.4), hw(2.0, -15.0, -15.0), hw(2.8, -16.0, -22.0)],
             [1.7, 1.1, 0.55]), 0.9)
    add(tube([hsurf(184, -20, 1.2), hw(-2.2, -14.5, -13.0), hw(-3.6, -15.4, -19.0)],
             [1.6, 1.0, 0.5]), 0.9)

    # Kapuzen-Kragen (Cowl): faellt vom Kopf auf die Schultern
    czz = [212, 219, 226, 232, 237, 241, 245]
    cfa = PchipInterpolator(czz, [32.0, 26.5, 19.5, 13.6, 10.8, 10.2, 10.6])
    cfb = PchipInterpolator(czz, [19.0, 18.0, 16.2, 13.8, 12.2, 11.6, 12.0])

    def cowl_ab(z):
        zc = np.clip(z, 212, 245)
        return cfa(zc), cfb(zc)

    CW = fold_specs(12, 23, amp=(1.0, 2.1), width=(0.062, 0.11), drift=0.12)
    cmod = fold_mod(212, 245, CW, lo=0.05, hi=1.0, pw=1.3)

    def cowlz(phi):
        o = 0.0
        for (p0, dr, amp, w) in CW:
            d = (phi - p0 + np.pi) % (2 * np.pi) - np.pi
            o = o + 2.0 * amp / 1.8 * np.exp(-(d / (w * 1.7)) ** 2)
        return 1.6 - o
    add(loft(212, 245, lambda z: cowl_ab(z)[0], lambda z: cowl_ab(z)[1],
             cy=lambda z: 2.2 + 0 * z, mod=cmod, zmod=cowlz, amp=2.2, rnd=1.2,
             rim=0.50, rimw=2.2), 2.4)
    add(capsule(v3(0, 1.4, 232.0), v3(0, 2.0, 244.0), 8.6, 8.0), 4.0)             # Hals
    log('Kopf fertig')

    # ------------------------------------------------- Kappe (Robin-Hood-Muetze)
    CAP_W0, CAP_TILT = 7.6, 0.17                      # Schnitthoehe, Neigung
    cr = np.array(HEAD_R) + np.array([2.4, 2.4, 1.2])

    def cap_fn():
        e_fn = ellipsoid(HEAD_C, tuple(cr), R_H)[0]

        def fn(X, Y, Z):
            lx, ly, lz = _local(X, Y, Z, HEAD_C, R_H)
            return smax(e_fn(X, Y, Z), (CAP_W0 + CAP_TILT * ly) - lz, 1.4)
        return fn, HEAD_C - 34, HEAD_C + 34

    add(custom(*cap_fn()), 2.0)
    # weiche, nach hinten sackende Haube
    add(ellipsoid(hw(0, -1.0, 9.0), (11.2, 12.8, 12.6), R_H), 5.0)
    add(ellipsoid(hw(0, -8.5, 12.0), (8.8, 9.8, 9.6), R_H), 5.0)
    add(ellipsoid(hw(0, -15.5, 11.0), (6.2, 7.0, 6.2), R_H), 4.5)
    add(ellipsoid(hw(0, -21.0, 7.5), (3.8, 4.6, 3.6), R_H), 4.0)
    sub(tube([hw(-8.5, -12.5, 14.6), hw(0, -14.5, 15.4), hw(8.5, -12.5, 14.6)], 0.62), 1.4)

    def cap_edge(phi_deg, off=0.0):
        """Punkt auf der unteren Kappenkante (Schnitt Ebene x Ellipsoid)."""
        lo_, hi_ = -40.0, 70.0
        for _ in range(40):
            el = (lo_ + hi_) / 2
            ph, e = math.radians(phi_deg), math.radians(el)
            d = np.array([math.sin(ph) * math.cos(e), math.cos(ph) * math.cos(e), math.sin(e)])
            pq = d / math.sqrt(float(np.sum((d / (cr + off)) ** 2)))
            if pq[2] - (CAP_W0 + CAP_TILT * pq[1]) > 0:
                hi_ = el
            else:
                lo_ = el
        ph, e = math.radians(phi_deg), math.radians((lo_ + hi_) / 2)
        d = np.array([math.sin(ph) * math.cos(e), math.cos(ph) * math.cos(e), math.sin(e)])
        pq = d / math.sqrt(float(np.sum((d / (cr + off)) ** 2)))
        return HEAD_C + pq[0] * U_H + pq[1] * V_H + pq[2] * W_H

    EDGE = [cap_edge(a_, 0.55) for a_ in range(-180, 181, 6)]
    add(tube(EDGE + [EDGE[0]], 1.65), 1.0)                                 # Rollrand
    sub(tube([cap_edge(a_, 2.9) for a_ in range(-180, 181, 6)] + [cap_edge(-180, 2.9)], 0.38), 0.3)
    for ang in (-62, 0, 62):                                               # Naehte der Kappe
        p0 = cap_edge(ang, 0.2)
        p1 = hw(0, -9.0, 14.5)
        sub(tube(catmull([tuple(p0), tuple((p0 + p1) / 2 + (p1 - HEAD_C) * 0.12), tuple(p1)], 10),
                 0.40), 0.3)
    # vordere Spitze (Schirm)
    pk = [(0.0, 7.4, 2.2), (1.8, 6.4, 1.8), (3.8, 5.0, 1.45), (5.8, 3.4, 1.1), (7.6, 1.8, 0.8)]
    e0 = cap_edge(0, 0.9)
    for (fo, ru, rw) in pk:
        add(ellipsoid(e0 + V_H * fo + W_H * (-fo * 0.16), (ru, 2.2, rw),
                      frame_uvw(U_H, V_H, W_H)), 1.4)

    # Feder: steckt am Rollrand und schwingt nach hinten
    fbase = cap_edge(58, 1.6)
    fpts = [fbase, fbase + U_H * 3.4 - V_H * 5.0 + W_H * 5.0,
            fbase + U_H * 5.6 - V_H * 14.0 + W_H * 7.0,
            fbase + U_H * 6.6 - V_H * 23.0 + W_H * 5.4,
            fbase + U_H * 6.4 - V_H * 30.5 + W_H * 1.0,
            fbase + U_H * 5.4 - V_H * 35.0 - W_H * 4.6]
    path = catmull([tuple(p) for p in fpts], 16)
    n = len(path)
    quill = []
    for i in range(n - 1):
        t = i / (n - 1)
        tan = unit(path[i + 1] - path[i])
        wdir = unit(np.cross(tan, U_H))
        Rf = np.array([tan, wdir, unit(np.cross(tan, wdir))])
        width = 4.6 * math.sin(math.pi * min(1, 0.06 + t * 1.02) ** 0.75) + 0.35
        c = (path[i] + path[i + 1]) / 2
        add(ellipsoid(c, (np.linalg.norm(path[i + 1] - path[i]) * 0.8, width, 0.62), Rf), 0.5)
        quill.append(path[i])
    quill.append(path[-1])
    add(tube(quill, np.linspace(1.15, 0.42, len(quill))), 0.45)            # Kiel
    for i in range(2, n - 2, 2):                                           # Fahnenkerben
        t = i / (n - 1)
        tan = unit(path[i + 1] - path[i - 1])
        wdir = unit(np.cross(tan, U_H))
        wd = 4.6 * math.sin(math.pi * min(1, 0.06 + t * 1.02) ** 0.75) + 0.35
        for sg in (-1, 1):
            p0 = path[i] + wdir * sg * 0.8
            p1 = path[i] + wdir * sg * wd * 1.04 - tan * 2.1
            sub(tube([p0, p1], 0.24), 0.12)
    for k in range(3):                                                     # Wicklung am Kiel
        add(torus(path[k + 1], unit(path[k + 2] - path[k]), 1.15, 0.34), 0.2)
    log('Muetze + Feder fertig')

    # --------------------------------------------------- Koecher + Gurt + Pfeile
    Q0, Q1 = v3(-34.0, 21.0, 104.0), v3(-17.0, 19.0, 187.0)
    qa = unit(Q1 - Q0)
    Rq = frame_from_axis(qa, hint=(0, -1, 0))
    add(capsule(Q0, Q1, 7.6, 8.2), 1.5)
    add(torus(Q1, qa, 8.4, 1.2), 0.8)                                     # Oberer Rand
    add(torus(Q0 + qa * 5, qa, 7.9, 1.0), 0.8)
    add(torus(Q0 + qa * 38, qa, 8.0, 0.9), 0.6)
    add(torus(Q0 + qa * 62, qa, 8.1, 0.9), 0.6)
    add(ellipsoid(Q0 - qa * 2.5, (6.4, 6.4, 3.6), Rq), 1.5)
    add(capsule(Q0 + qa * 6 + Rq[0] * 7.6, Q0 + qa * 70 + Rq[0] * 8.1, 0.9), 0.4)   # Naht
    shafts = [(-2.4, 1.2, 38, 0.2), (2.2, 2.6, 30, -0.3), (0.0, -3.0, 44, 0.1), (-4.0, -2.0, 33, -0.1),
              (4.0, -1.2, 40, 0.15), (-1.0, 3.8, 27, 0.25), (1.5, -4.4, 35, -0.2)]
    for (ox, oy, ext, tw) in shafts:
        base = Q1 + Rq[0] * ox + Rq[1] * oy - qa * 10
        dirv = unit(qa + Rq[0] * ox * 0.012 + Rq[1] * oy * 0.012)
        tip = Q1 + Rq[0] * ox + Rq[1] * oy + dirv * ext
        add(capsule(base, tip, 1.1, 1.0), 0.3)
        # Befiederung
        Rv = frame_from_axis(dirv, hint=Rq[0])
        for k in range(3):
            ang = math.radians(90 + 120 * k + tw * 40)
            nrm = Rv[0] * math.cos(ang) + Rv[1] * math.sin(ang)
            tng = unit(np.cross(dirv, nrm))
            cvane = tip - dirv * 8.5 + nrm * 2.3
            Rvane = np.array([tng, nrm, dirv])
            add(rbox(cvane, (0.55, 2.4, 6.5), Rvane, 0.5), 0.3)
        add(ellipsoid(tip + dirv * 0.4, (1.5, 1.5, 1.5)), 0.2)
    # Schultergurt (liegt auf der Wams-Oberflaeche)
    log('Koecher fertig')

    # Gurt quer ueber die Brust: Offset-Schale um den Koerper
    bpts = np.array([[-19, 213], [-9, 199], [2, 185], [12.5, 170], [20.5, 156]], float)
    bpoly = catmull([(x, 0, z) for x, z in bpts], 20)[:, [0, 2]]

    def band_fn(pts2, hw_, th, ymax):
        def f(X, Y, Z):
            X_, Z_ = np.broadcast_arrays(X, Z)
            dmin = np.full(np.broadcast(X, Y, Z).shape, 1e9, np.float32)
            for (px, pz), (qx, qz) in zip(pts2[:-1], pts2[1:]):
                ax, az = qx - px, qz - pz
                L2 = ax * ax + az * az
                t = np.clip(((X_ - px) * ax + (Z_ - pz) * az) / L2, 0, 1)
                dd = np.sqrt((X_ - px - t * ax) ** 2 + (Z_ - pz - t * az) ** 2)
                dmin = np.minimum(dmin, dd)
            return np.maximum(dmin - hw_, Y - ymax)
        mn = pts2.min(0) - hw_ - 2
        mx = pts2.max(0) + hw_ + 2
        return f, v3(mn[0], -40, mn[1]), v3(mx[0], 40, mx[1])

    fb_, bmn, bmx = band_fn(bpoly, 2.9, 1.4, 4.0)
    fig_f_before = fig.f.copy()
    sl_i0 = np.maximum(np.floor((bmn - 4 - fig.lo) / fig.h).astype(int), 0)
    sl_i1 = np.minimum(np.ceil((bmx + 4 - fig.lo) / fig.h).astype(int) + 1, fig.n)
    xs = (fig.lo[0] + fig.h * np.arange(sl_i0[0], sl_i1[0])).astype(F32)[:, None, None]
    ys = (fig.lo[1] + fig.h * np.arange(sl_i0[1], sl_i1[1])).astype(F32)[None, :, None]
    zs = (fig.lo[2] + fig.h * np.arange(sl_i0[2], sl_i1[2])).astype(F32)[None, None, :]
    sl = (slice(sl_i0[0], sl_i1[0]), slice(sl_i0[1], sl_i1[1]), slice(sl_i0[2], sl_i1[2]))
    base = fig_f_before[sl]
    strap = np.maximum(fb_(xs, ys, zs), base - 1.5)
    fig.f[sl] = smin(fig.f[sl], strap.astype(F32), 0.5).astype(F32)
    del fig_f_before
    # Gurt ueber die rechte Schulter nach hinten
    add(capsule(v3(-19, -3, 214), v3(-20, 8, 226), 2.4), 0.6)
    add(capsule(v3(-20, 8, 226), v3(-15, 17, 192), 2.4, 2.0), 0.6)
    # Schnalle am Gurt
    add(rbox(v3(14.0, -14.2, 166.0), (3.3, 1.2, 3.8), None, 0.5), 0.4)
    sub(rbox(v3(14.0, -14.9, 166.0), (1.8, 1.5, 2.3), None, 0.3), 0.2)
    log('Gurt fertig')

    # ----------------------------------------------- Brustschnuerung am Wams
    ys1 = fig.surface_y(0.0, 210, -40, 10, 0.1)
    if ys1 is not None:                                      # erst der Schlitz ...
        sub(capsule(v3(0, ys1 - 0.3, 211), v3(0, ys1 - 0.3, 160), 0.5), 0.2)
    for i in range(6):                                       # ... dann die Schnuerung darueber
        zc = 205 - i * 8.2
        ysf = fig.surface_y(0.0, zc, -40, 10, 0.1)
        if ysf is None:
            continue
        add(capsule(v3(-3.4, ysf - 0.15, zc + 1.9), v3(3.4, ysf - 0.35, zc - 1.9), 0.78), 0.35)
        add(capsule(v3(3.4, ysf - 0.35, zc + 1.9), v3(-3.4, ysf - 0.15, zc - 1.9), 0.78), 0.35)
        add(ellipsoid(v3(0, ysf - 0.25, zc), (1.0, 0.9, 1.0)), 0.5)   # Kreuzungspunkt

    add_details(fig)
    log('Naehte / Beschlaege fertig')
    surface_texture(fig)
    log('Oberflaechenstruktur fertig')

    # ----------------------------------------------------------- Boden kappen + Zapfen
    sub(rbox(v3(0, 0, -30.0), (200, 200, 30.0), None, 0.0), 0.0)
    for (px, py) in peg_positions():
        add(cylinder(v3(px, py, -PEG_LEN), v3(px, py, 0.6), PEG_R, 0.6), 0.5)
    log('Figur: Feld fertig')
    return fig


# ================================================= Naehte, Beschlaege, Struktur
def shoe(origin, d, u, sole_z, stations, grow=0.0, hcap=None, rad=2.0, prof='egg', n=2.5):
    """Glatte Schuhform: Querschnitt (abgerundetes Rechteck) laeuft entlang der Zehenrichtung."""
    oo = np.array([t[0] for t in stations], float)
    fw_ = PchipInterpolator(oo, np.array([t[1] for t in stations]) + grow)
    fh_ = PchipInterpolator(oo, [t[2] for t in stations])
    d = unit(d); u = unit(u)
    o = np.asarray(origin, float)

    def fn(X, Y, Z):
        px, py, pz = X - o[0], Y - o[1], Z - o[2]
        t = px * d[0] + py * d[1] + pz * d[2]
        q = px * u[0] + py * u[1] + pz * u[2]
        zz = Z - sole_z
        tc = np.clip(t, oo[0], oo[-1])
        W = fw_(tc)
        H = fh_(tc) if hcap is None else np.minimum(fh_(tc), hcap)
        if prof == 'egg':                      # fussfoermiger Querschnitt
            A, B, zc = W, H * 0.57, H * 0.50
            e = (np.abs(q) / A) ** n + (np.abs(zz - zc) / B) ** n
            dd = (np.maximum(e, 1e-9) ** (1.0 / n) - 1.0) * np.minimum(A, B)
        else:                                   # flache Platte
            r = np.minimum(rad, np.minimum(W, H / 2) * 0.8)
            a = np.abs(q) - np.maximum(W - r, 0.05)
            b = np.abs(zz - H / 2) - np.maximum(H / 2 - r, 0.05)
            dd = (np.sqrt(np.maximum(a, 0) ** 2 + np.maximum(b, 0) ** 2)
                  + np.minimum(np.maximum(a, b), 0) - r)
        return smax(dd, np.maximum(oo[0] - t, t - oo[-1]), 1.2)
    m = float(max(np.abs(oo).max(), np.max([t[1] for t in stations]) + grow,
                  np.max([t[2] for t in stations]))) + 3
    return fn, o - m, o + m


def ring_path(a, b, z, n=160, cy=0.0, phi0=-math.pi, phi1=math.pi):
    ph = np.linspace(phi0, phi1, n)
    return np.stack([a * np.cos(ph), cy + b * np.sin(ph), np.full(n, float(z))], 1)


def add_details(fig):
    add, sub = fig.add, fig.sub
    IN_AX = lambda P: np.stack([-P[:, 0], -(P[:, 1] - 1.0), np.zeros(len(P))], 1)   # zur Koerperachse

    # ---- Guertel: Steppnaht oben und unten, Loecher, Riemenende
    for z in (147.6, 152.0):
        P = ring_path(24.5, 18.5, z)
        stitches(fig, P, IN_AX(P), pitch=2.0, length=1.0, r=0.30, out=0.07)
    for i in range(6):                                            # Loecher im Riemen
        x = 7.5 + 3.4 * i
        p = np.array([[x, -22.0, 149.8]])
        hit, nrm = project(fig, p, np.array([[0, 1.0, 0]]))
        if len(hit):
            sub(capsule(hit[0] + nrm[0] * 0.6, hit[0] - nrm[0] * 1.6, 0.52), 0.2)
    add(tube([v3(-5.0, -19.6, 148.6), v3(-9.5, -19.0, 146.0), v3(-11.5, -18.4, 141.0),
              v3(-11.0, -17.6, 136.5)], [2.3, 2.2, 2.0, 1.6]), 0.6)       # haengendes Riemenende
    add(rbox(v3(-10.8, -18.2, 137.0), (1.9, 1.0, 1.2), None, 0.4), 0.3)

    # ---- Wams: Halsausschnitt, Schulter- und Seitennaehte
    for sg in (-1, 1):
        P = catmull([(sg * 2.0, -17.0, 212.0), (sg * 7.5, -16.0, 203.0),
                     (sg * 6.0, -17.4, 188.0), (sg * 3.0, -18.6, 172.0), (0.0, -19.0, 162.0)], 12)
        seam(fig, P, np.array([[0, 1.0, 0]]), r=0.46, sub=False, pitch=0.6)       # Besatzkante
        stitches(fig, P, np.array([[0, 1.0, 0]]), pitch=2.2, length=1.0, r=0.24, out=0.10)
        Q = catmull([(sg * 13.0, -8.0, 216.0), (sg * 21.0, -4.0, 210.0), (sg * 25.5, 2.0, 203.0)], 10)
        seam(fig, Q, -np.stack([Q[:, 0], Q[:, 1] - 1, np.zeros(len(Q)) - 1.2], 1), r=0.34)
        Sd = catmull([(sg * 26.5, 1.0, 198.0), (sg * 25.0, 1.0, 182.0), (sg * 22.5, 1.0, 166.0),
                      (sg * 22.0, 1.0, 154.0)], 10)
        seam(fig, Sd, -np.stack([Sd[:, 0], Sd[:, 1] - 1, np.zeros(len(Sd))], 1), r=0.30)

    # ---- Armschiene: Naht und Nieten
    ax = unit(W_L - E_L)
    side = unit(np.cross(ax, v3(0, 0, 1)))
    nrm = unit(np.cross(side, ax))
    for sg in (-1, 1):
        P = np.array([E_L + (W_L - E_L) * t + side * sg * 6.0 for t in np.linspace(0.33, 0.90, 10)])
        stitches(fig, P, -np.repeat(side[None] * sg, len(P), 0), pitch=1.9, length=0.9, r=0.26, out=0.08)
    studs(fig, np.array([E_L + (W_L - E_L) * t + nrm * 6.5 for t in (0.36, 0.52, 0.68, 0.84)]),
          -np.repeat(nrm[None], 4, 0), r=0.62, out=0.30)

    # ---- Koecher: Riemen, Schnallen, Nieten, Naht
    Q0, Q1 = v3(-34.0, 21.0, 104.0), v3(-17.0, 19.0, 187.0)
    qa = unit(Q1 - Q0)
    Rq = frame_from_axis(qa, hint=(0, -1, 0))
    for t in (0.22, 0.74):
        c = Q0 + (Q1 - Q0) * t
        r_ = 7.7 + 0.5 * t
        P = np.array([c + Rq[0] * r_ * math.cos(a) + Rq[1] * r_ * math.sin(a)
                      for a in np.linspace(-math.pi, math.pi, 90)])
        add(tube(np.vstack([P, P[:1]]), 1.05), 0.5)
        bp = c + Rq[1] * (r_ + 0.6)
        add(rbox(bp, (2.6, 1.0, 2.6), frame_uvw(Rq[0], Rq[1], qa), 0.5), 0.4)
        sub(rbox(bp + Rq[1] * 0.7, (1.4, 1.0, 1.4), frame_uvw(Rq[0], Rq[1], qa), 0.3), 0.2)
    for t in np.linspace(0.12, 0.92, 9):                       # Naht laengs
        pass
    P = np.array([Q0 + (Q1 - Q0) * t + Rq[0] * (7.7 + 0.5 * t) for t in np.linspace(0.08, 0.95, 24)])
    stitches(fig, P, -np.repeat(Rq[0][None], len(P), 0), pitch=2.1, length=1.0, r=0.28, out=0.08)
    studs(fig, np.array([Q0 + (Q1 - Q0) * t + Rq[1] * 8.0 for t in (0.14, 0.33, 0.52, 0.70, 0.88)]),
          -np.repeat(Rq[1][None], 5, 0), r=0.60, out=0.28)

    # ---- Schultergurt: Steppnaht beidseits
    bp = catmull([(-19, 0, 213), (-9, 0, 199), (2, 0, 185), (12.5, 0, 170), (20.5, 0, 156)], 14)
    for sg in (-1, 1):
        tg = np.gradient(bp, axis=0)
        tg /= np.linalg.norm(tg, axis=1, keepdims=True)
        off = np.cross(tg, np.array([0, 1.0, 0])) * sg * 2.0
        P = bp + off
        stitches(fig, P, np.array([[0, 1.0, 0]]), pitch=2.0, length=0.9, r=0.26, out=0.06)

    # ---- Spannfalten am Wams (vorn und hinten)
    for sg in (-1, 1):
        for (a0, a1, a2, rr) in (((21, 201), (14, 185), (5, 169), 0.44),
                                 ((25, 193), (17, 177), (8, 164), 0.40),
                                 ((13, 207), (10, 193), (5, 179), 0.34)):
            P = catmull([(sg * a0[0], -34.0, a0[1]), (sg * a1[0], -34.0, a1[1]),
                         (sg * a2[0], -34.0, a2[1])], 12)
            seam(fig, P, np.array([[0, 1.0, 0]]), r=rr, reach=56.0)
            Q = catmull([(sg * a0[0], 34.0, a0[1] - 2), (sg * a1[0], 34.0, a1[1] - 2),
                         (sg * a2[0], 34.0, a2[1] - 2)], 12)
            seam(fig, Q, np.array([[0, -1.0, 0]]), r=rr * 0.9, reach=56.0)
    for z in (157.0, 160.4, 163.6):                               # Raffung ueber dem Guertel
        for (q0, q1) in ((-math.pi, -math.pi / 3), (-math.pi / 3, math.pi / 3),
                         (math.pi / 3, math.pi)):
            P = ring_path(27.0, 21.0, z, n=48, cy=1.0, phi0=q0, phi1=q1)
            seam(fig, P, IN_AX(P), r=0.32, reach=26.0)
    for sg in (-1, 1):                                            # Falten unter den Achseln
        for k in range(3):
            P = catmull([(sg * (23 - 1.5 * k), -22.0, 204 - 5 * k),
                         (sg * (27 - 1.0 * k), -4.0, 199 - 5 * k),
                         (sg * (25 - 1.0 * k), 12.0, 196 - 5 * k)], 10)
            seam(fig, P, -np.stack([P[:, 0], P[:, 1] - 1, np.zeros(len(P))], 1),
                 r=0.34, reach=30.0)

    # ---- Muetze: Naht am Rollrand, Federkiel-Wicklung
    P = np.array([hsurf(a, 20.0, 0.6) for a in np.linspace(-180, 175, 72)])
    stitches(fig, P, np.stack([HEAD_C - p for p in P]), pitch=2.3, length=1.0, r=0.26, out=0.10)
    # ---- Beutel: Klappennaht + Knopf
    P = catmull([(-29.0, -9.0, 145.6), (-24.0, -13.6, 145.2), (-18.6, -9.0, 145.4)], 10)
    stitches(fig, P, np.array([[0.4, 1.0, 0.2]]), pitch=1.8, length=0.8, r=0.26, out=0.06)


def surface_texture(fig):
    """Feine Struktur: Stoffgewebe, Ledernarbung, gleichmaessige Mikro-Rauheit."""
    cloth = mask_of([
        rbox(v3(0, 1, 128), (40, 30, 26), None, 2.0),            # Rock
        rbox(v3(0, 1, 156), (26, 20, 7), None, 2.0),             # Bauschung
        rbox(v3(0, 0, 231), (34, 22, 19), None, 2.0),            # Kragen
        capsule(v3(0, -2, 252), v3(0, -2, 276), 24.0),           # Kopftuch / Muetze
        capsule(S_L, S_L + (E_L - S_L) * 0.55, 14.0),
        capsule(S_R, S_R + (E_R - S_R) * 0.55, 14.0),            # Aermel
        capsule(HIP[1], KNEE[1], 15.0), capsule(HIP[-1], KNEE[-1], 15.0),
        capsule(KNEE[1], ANK[1] + v3(0, 0, 14), 12.0),
        capsule(KNEE[-1], ANK[-1] + v3(0, 0, 14), 12.0),
    ], soft=3.0)
    leather = mask_of([
        rbox(v3(0, 1, 149.8), (26, 20, 4.6), None, 1.0),         # Guertel
        capsule(v3(-34, 21, 100), v3(-17, 19, 190), 9.6),        # Koecher
        capsule(E_L + (W_L - E_L) * 0.28, W_L, 7.6),             # Armschiene
        capsule(ANK[1] + v3(0, 0, -12), ANK[1] + v3(0, 0, 50), 11.0),
        capsule(ANK[-1] + v3(0, 0, -12), ANK[-1] + v3(0, 0, 50), 11.0),
        ellipsoid(v3(-23.5, -7.5, 143.5), (7.5, 8.5, 9.5)),      # Beutel
    ], soft=2.0)
    fig.displace(0.30, 7.0, mask=cloth, band=1.6, seed=11, octaves=3, gain=0.52)
    fig.displace(0.085, 1.5, mask=cloth, band=1.2, seed=12, octaves=2, gain=0.5)
    fig.displace(0.10, 1.3, mask=leather, band=1.2, seed=13, octaves=3, gain=0.55, ridged=True)
    fig.displace(0.045, 2.2, band=1.0, seed=14, octaves=2, gain=0.5)


# =============================================================== BOGEN
BOW_ZG = F_L[2]
BOW_XG = F_L[0]
BOW_HALF = 128.0       # Halbspannweite (Griffmitte bis Spitze)
BOW_RECURVE = 26.0     # Auslenkung der Spitzen Richtung Schuetze (gespannt)
FIST_HH = 7.6          # halbe Faustoehe


def bow_x(dz):
    """x-Lage der Wirbelsaeule des Bogens in Abhaengigkeit vom Abstand zur Griffmitte."""
    t = np.clip((abs(dz) - 12.0) / (BOW_HALF - 12.0), 0, 1)
    return BOW_XG - BOW_RECURVE * (t ** 2.0)


def limb_shape(sign):
    """Glatter Wurfarm: Abstand zur Mittellinie (KD-Baum) + Profil (abgerundetes Rechteck)."""
    from scipy.spatial import cKDTree
    dzs = np.linspace(FIST_HH + 8.0, BOW_HALF, 600)
    P = np.stack([bow_x(dzs), BOW_ZG + sign * dzs], 1)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    tt = (dzs - dzs[0]) / (dzs[-1] - dzs[0])
    thick = 3.8 - 1.7 * tt         # in der Bogenebene
    wid = 5.6 - 2.3 * tt           # quer dazu (y)
    tree = cKDTree(P)
    last = len(P) - 1

    def fn(X, Y, Z):
        shp = np.broadcast(X, Y, Z).shape
        q = np.stack([np.broadcast_to(X, shp).ravel(), np.broadcast_to(Z, shp).ravel()], 1)
        yy = np.broadcast_to(Y, shp).ravel() - BOW_Y
        _, idx = tree.query(q)
        dx = q[:, 0] - P[idx, 0]
        dz = q[:, 1] - P[idx, 1]
        n = dx * N[idx, 0] + dz * N[idx, 1]
        al = dx * T[idx, 0] + dz * T[idx, 1]
        r = 0.95
        qa = np.abs(n) - thick[idx] / 2 + r
        qb = np.abs(yy) - wid[idx] / 2 + r
        d = np.sqrt(np.maximum(qa, 0) ** 2 + np.maximum(qb, 0) ** 2) + np.minimum(np.maximum(qa, qb), 0) - r
        ext = np.where(idx == 0, -al, np.where(idx == last, al, -1e3))
        d = np.maximum(d, ext)
        return d.reshape(shp).astype(np.float32)
    zlo_, zhi_ = (BOW_ZG + FIST_HH, BOW_ZG + BOW_HALF + 4) if sign > 0 else (BOW_ZG - BOW_HALF - 4, BOW_ZG - FIST_HH)
    return fn, v3(BOW_XG - BOW_RECURVE - 6, BOW_Y - 6, zlo_), v3(BOW_XG + 6, BOW_Y + 6, zhi_)


def build_bow_half(sign, h):
    """sign=+1 oberer Teil (inkl. Sehnenhaelfte), -1 unterer Teil."""
    lo = (NOCK[0] - 6, BOW_Y - 12, BOW_ZG - 12 if sign > 0 else BOW_ZG - BOW_HALF - 8)
    hi = (BOW_XG + 8, BOW_Y + 12, BOW_ZG + BOW_HALF + 8 if sign > 0 else BOW_ZG + 12)
    f = Field(lo, hi, h)
    # Zapfen (steckt in der Bohrung der Faust)
    # Griff: Zapfen durch die Faust + Lederwicklung darueber
    z1 = BOW_ZG + sign * (FIST_HH + 1.2)
    f.add(cylinder(v3(BOW_XG, BOW_Y, BOW_ZG + sign * 0.3), v3(BOW_XG, BOW_Y, z1), GRIP_R, 0.4), 0.3)
    z2 = BOW_ZG + sign * (FIST_HH + 13.0)
    f.add(cylinder(v3(BOW_XG, BOW_Y, z1), v3(BOW_XG, BOW_Y, z2), 3.45, 0.6), 0.5)
    for k in range(5):                                   # ueberlappende Wicklung
        zc = BOW_ZG + sign * (FIST_HH + 1.9 + 2.35 * k)
        f.add(torus(v3(BOW_XG, BOW_Y, zc), v3(0, 0, 0.06 * sign + 1), 3.30, 0.62), 0.25)
    f.add(torus(v3(BOW_XG, BOW_Y, z2 - sign * 0.6), v3(0, 0, 1), 3.5, 0.7), 0.4)
    f.sub(torus(v3(BOW_XG, BOW_Y, z2 + sign * 1.4), v3(0, 0, 1), 4.3, 1.4), 0.5)
    # Wurfarm
    f.add(limb_shape(sign), 1.2)
    # Hornnocke an der Spitze
    tip = v3(bow_x(BOW_HALF), BOW_Y, BOW_ZG + sign * BOW_HALF)
    tdir = unit(tip - v3(bow_x(BOW_HALF - 14), BOW_Y, BOW_ZG + sign * (BOW_HALF - 14)))
    f.add(capsule(tip - tdir * 7.0, tip + tdir * 1.2, 2.25, 2.55), 1.2)
    f.add(ellipsoid(tip + tdir * 1.0, (2.5, 2.6, 2.5)), 0.8)
    sdir_ = unit(NOCK - tip)
    f.sub(capsule(tip + tdir * 1.4 + sdir_ * 1.0, tip + tdir * 1.4 - sdir_ * 4.0, 0.62), 0.3)
    # Sehnenhaelfte bis zum Nockpunkt, mit Mittelwicklung
    f.add(capsule(tip + sdir_ * 1.6, NOCK, 0.92, 0.92), 0.4)
    if sign > 0:
        f.add(ellipsoid(NOCK, (1.45, 1.45, 1.45)), 0.3)
    for k in range(7):                                           # Wicklung am Nockpunkt
        p = NOCK + sdir_ * (-12.6 + 1.9 * k)
        f.add(torus(p, sdir_, 1.12, 0.30), 0.18)
    f.displace(0.055, 2.6, band=0.9, seed=31 + sign, octaves=2, gain=0.5)   # Holzmaserung
    return f


# ================================================================ PFEIL
def build_arrow(h):
    x0, x1 = NOCK[0] - 3.5, ARROW_TIP + 2.0
    lo = (x0 - 2, ARROW_Y - 10, ARROW_Z - 10)
    hi = (x1 + 2, ARROW_Y + 10, ARROW_Z + 10)
    f = Field(lo, hi, h)
    c0 = v3(NOCK[0] - 2.5, ARROW_Y, ARROW_Z)
    f.add(capsule(c0, v3(ARROW_TIP - 18.5, ARROW_Y, ARROW_Z), 1.38, 1.30), 0.2)
    f.add(ellipsoid(c0, (2.2, 1.9, 1.9)), 0.4)                       # Nock-Ende
    f.sub(rbox(c0 + v3(-1.4, 0, 0), (1.2, 0.35, 3.0), None, 0.1), 0.1)  # Kerbe fuer die Sehne
    # Befiederung: drei Federn (eine nach unten, zwei schraeg nach oben)
    fl0, fl1 = NOCK[0] + 2.5, NOCK[0] + 21.0
    for ang_deg in (0, 120, 240):
        ang = math.radians(ang_deg)
        nrm = v3(0, math.cos(ang), math.sin(ang))
        tng = v3(1, 0, 0)
        R = np.array([tng, nrm, np.cross(tng, nrm)])
        for i in range(32):
            t = i / 31
            xc = fl0 + (fl1 - fl0) * t
            hgt = 2.9 * (0.35 + 0.65 * math.sin(math.pi * min(1, 0.16 + 0.84 * t))) * (1 - 0.55 * t ** 3)
            cc = v3(xc, ARROW_Y, ARROW_Z) + nrm * (hgt / 2 + 0.45)
            f.add(rbox(cc, (1.1, hgt / 2 + 0.45, 0.42), R, 0.32), 0.22)
        for i in range(9):                                      # Fahnenkerben
            t = 0.18 + 0.08 * i
            xc = fl0 + (fl1 - fl0) * t
            hgt = 2.9 * (0.35 + 0.65 * math.sin(math.pi * min(1, 0.16 + 0.84 * t)))
            f.sub(capsule(v3(xc, ARROW_Y, ARROW_Z) + nrm * 1.0,
                          v3(xc - 1.0, ARROW_Y, ARROW_Z) + nrm * (hgt + 0.6), 0.16), 0.1)
    for xb in (fl0 - 0.8, fl1 + 0.8):                            # Bindegarn
        f.add(torus(v3(xb, ARROW_Y, ARROW_Z), v3(1, 0, 0), 1.42, 0.30), 0.15)
    # Spitze: Blattspitze
    f.add(capsule(v3(ARROW_TIP - 20.5, ARROW_Y, ARROW_Z), v3(ARROW_TIP - 15.5, ARROW_Y, ARROW_Z),
                  1.45, 1.95), 0.3)
    pts = [(ARROW_TIP - 15.5, 0.0), (ARROW_TIP - 12.0, 3.0), (ARROW_TIP - 6.5, 3.4), (ARROW_TIP - 0.3, 0.0)]
    for (xa, wa), (xb, wb) in zip(pts[:-1], pts[1:]):
        f.add(rbox(v3((xa + xb) / 2, ARROW_Y, ARROW_Z), ((xb - xa) / 2 + 0.3, max(wa, wb) / 2 + 0.2, 0.9),
                   None, 0.3), 0.2)
    f.add(capsule(v3(ARROW_TIP - 6.5, ARROW_Y, ARROW_Z), v3(ARROW_TIP + 0.1, ARROW_Y, ARROW_Z),
                  1.2, 0.25), 0.2)
    for xr in (NOCK[0] + 29.0, NOCK[0] + 31.2):                      # Zierringe
        f.add(torus(v3(xr, ARROW_Y, ARROW_Z), v3(1, 0, 0), 1.35, 0.35), 0.2)
    return f


# ================================================================ SOCKEL
def _slab(w0, d0, z0, w1, d1, z1):
    """Abgeschrägte Platte (Konvexe Huelle zweier Rechtecke)."""
    pts = []
    for (w, d, z) in ((w0, d0, z0), (w1, d1, z1)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                pts.append((sx * w / 2, sy * d / 2, z))
    return trimesh.convex.convex_hull(np.array(pts, float))


def build_base_mesh():
    """Sockel als exakte CSG-Konstruktion (Manifold): wenig Dreiecke, garantiert wasserdicht."""
    from shapely.geometry import Polygon
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    import matplotlib
    W, D = 140.0, 72.0
    ch = 1.0
    layers = [
        _slab(W - 2 * 0.6, D - 2 * 0.6, -30.0, W, D, -29.4),                # Fuss (unten leicht gefast)
        _slab(W, D, -29.4, W, D, -26.0),
        _slab(W, D, -26.0, W - 2 * ch, D - 2 * ch, -25.0),                    # obere Fase des Fusses
        _slab(W - 6, D - 6, -25.0, W - 14, D - 14, -21.8),                    # Kehle (Karnies unten)
        _slab(W - 14, D - 14, -21.8, W - 14, D - 14, -8.2),                   # Schaft
        _slab(W - 14, D - 14, -8.2, W - 6, D - 6, -5.0),                      # Kehle (Gesims)
        _slab(W - 6, D - 6, -5.0, W - 4, D - 4, -4.4),
        _slab(W - 4, D - 4, -4.4, W - 4, D - 4, -0.9),                        # Deckplatte
        _slab(W - 4, D - 4, -0.9, W - 6, D - 6, 0.0),
    ]
    base = trimesh.boolean.union(layers, engine='manifold')
    yf = -(D - 14) / 2        # Vorderseite des Schafts
    xf = (W - 14) / 2
    cuts = []
    # eingelassene Fuellungen (Paneele) rundum
    pz0, pz1 = -20.4, -9.6
    cuts.append(trimesh.creation.box(extents=[W - 14 - 14, 2.0, pz1 - pz0], transform=trimesh.transformations.translation_matrix([0, yf + 0.0, (pz0 + pz1) / 2])))
    cuts[-1].apply_translation([0, 0, 0])
    cuts.append(trimesh.creation.box(extents=[W - 14 - 14, 2.0, pz1 - pz0], transform=trimesh.transformations.translation_matrix([0, -yf, (pz0 + pz1) / 2])))
    for sx in (-1, 1):
        cuts.append(trimesh.creation.box(extents=[2.0, D - 14 - 14, pz1 - pz0],
                                         transform=trimesh.transformations.translation_matrix([sx * xf, 0, (pz0 + pz1) / 2])))
    # Schrift
    fpath = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSerif-Bold.ttf')
    fp = FontProperties(fname=fpath)
    text = 'THE GREEN ARCHER  \u00b7  1940'
    tp = TextPath((0, 0), text, size=5.6, prop=fp)
    polys = [Polygon(pg) for pg in tp.to_polygons() if len(pg) > 2]
    geom = None
    for pg in sorted(polys, key=lambda q: -q.area):
        geom = pg if geom is None else geom.symmetric_difference(pg)
    minx, miny, maxx, maxy = geom.bounds
    from shapely import affinity
    geom = affinity.translate(geom, -(minx + maxx) / 2, -(miny + maxy) / 2)
    depth = 1.6
    H = depth + 0.4
    gl = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
    for pg in gl:
        if pg.area < 0.05:
            continue
        ex = trimesh.creation.extrude_polygon(pg, H)
        # (u,v,w) -> (u,-w,v)  (Drehung um +90 Grad um x)
        Rm = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        ex.apply_transform(Rm)
        ex.apply_translation([0, yf + depth, (pz0 + pz1) / 2])
        cuts.append(ex)
    # Sockel-Buchsen fuer die Fusszapfen (0.25 mm Spiel)
    for (px_, py_) in peg_positions():
        cyl = trimesh.creation.cylinder(radius=PEG_R + 0.25, height=PEG_LEN + 1.4, sections=48,
                                        transform=trimesh.transformations.translation_matrix([px_, py_, -PEG_LEN / 2 + 0.7 - 0.3]))
        cuts.append(cyl)
    base = trimesh.boolean.difference([base] + cuts, engine='manifold')
    return base


# ================================================================== MAIN
def carve(part, fig, gap=0.12):
    """Entfernt aus `part` alles, was in die Figur (plus Spiel `gap`) hineinragt -> sauberer Kontakt."""
    xs, ys, zs = part.axes()
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    for k0 in range(0, len(zs), 24):
        zk = zs[k0:k0 + 24]
        P = np.stack([np.repeat(X[:, :, None], len(zk), 2), np.repeat(Y[:, :, None], len(zk), 2),
                      np.broadcast_to(zk, X.shape + (len(zk),))], -1).reshape(-1, 3)
        fs = fig.sample(P).reshape(X.shape + (len(zk),))
        blk = part.f[:, :, k0:k0 + 24]
        part.f[:, :, k0:k0 + 24] = np.maximum(blk, (gap - fs).astype(np.float32))


def clearance_report(fig):
    segs = {}
    for sg in (1, -1):
        tip = v3(bow_x(BOW_HALF), BOW_Y, BOW_ZG + sg * BOW_HALF)
        segs['Sehne ' + ('oben' if sg > 0 else 'unten')] = (tip + v3(-1.8, 0, -sg * 0.8), NOCK, 0.95)
    segs['Pfeilschaft'] = (v3(NOCK[0] - 2.5, ARROW_Y, ARROW_Z),
                           v3(ARROW_TIP - 18.5, ARROW_Y, ARROW_Z), 1.35)
    out = {}
    for k, (a, b, r) in segs.items():
        t = np.linspace(0, 1, 400)[:, None]
        pts = a + (b - a) * t
        out[k] = float(fig.sample(pts).min() - r)
        log(f'  Abstand {k}: {out[k]:+.2f} mm zur Figur (negativ = Ueberschneidung)')
    return out


def defects(m):
    """(offene Kanten, Kanten mit mehr als zwei Flaechen)"""
    e = np.sort(m.edges, axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum())


def tidy(m):
    """Doppelte/entartete Dreiecke entfernen - noetig, damit die Vereinfachung dicht bleibt."""
    m.merge_vertices(digits_vertex=5)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_normals(m)
    return m


def finish(f, name, outdir, target_faces=None, min_vol=1.0):
    m = f.mesh()
    log(name, 'Marching Cubes:', len(m.faces), 'Dreiecke')
    m = clean_mesh(m, min_vol)
    comps = m.split(only_watertight=False)
    log(name, 'roh wasserdicht:', m.is_watertight, ' Komponenten:', len(comps))
    if len(comps) > 1:
        for c in comps:
            log('   Komponente: Volumen %.1f mm3, Dreiecke %d, bounds %s' % (abs(c.volume), len(c.faces), np.round(c.bounds, 0).tolist()))
    tidy(m)
    if target_faces and len(m.faces) > target_faces:
        import fast_simplification as fs
        v, fcs = fs.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                             target_reduction=1 - target_faces / len(m.faces), agg=3)
        m2 = trimesh.Trimesh(v, fcs, process=True)
        tidy(m2)
        b2, n2 = defects(m2)
        log(name, 'vereinfacht auf', len(m2.faces), 'Dreiecke; offene Kanten:', b2,
            ' Mehrfachkanten:', n2)
        if b2 > 0:
            m3 = close_mesh(m2.copy())
            b3, n3 = defects(m3)
            log(name, 'nach Reparatur: offene Kanten:', b3, ' Mehrfachkanten:', n3)
            if b3 == 0:
                m2, b2, n2 = m3, b3, n3
        if b2 == 0:
            m = m2
    b, n = defects(m)
    log(name, 'final:', len(m.faces), 'Dreiecke, offene Kanten:', b, ' Mehrfachkanten:', n,
        ' wasserdicht:', m.is_watertight)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=float, default=0.4)
    ap.add_argument('--parts', default='figur,bogen,pfeil,sockel')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stl'))
    ap.add_argument('--faces', type=int, default=700000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = a.parts.split(',')
    hb = min(a.h, 0.18)
    meshes = {}

    def save(name, m):
        meshes[name] = m
        m.export(os.path.join(a.out, f'{name}_assembled.stl'))
        log(name, 'watertight=', m.is_watertight, 'Volumen=%.0f mm3' % m.volume, 'bounds', np.round(m.bounds, 1).tolist())

    fig = build_figure(a.h) if ('figur' in parts or 'bogen' in parts or 'pfeil' in parts) else None
    if fig is not None:
        clearance_report(fig)
    if 'figur' in parts:
        save('figur', finish(fig, 'figur', a.out, a.faces, 800.0))
    if 'bogen' in parts:
        for sg, nm in ((1, 'bogen_oben'), (-1, 'bogen_unten')):
            bf = build_bow_half(sg, hb)
            carve(bf, fig)
            save(nm, finish(bf, nm, a.out, 300000, 60.0))
    if 'pfeil' in parts:
        af = build_arrow(hb)
        carve(af, fig)
        save('pfeil', finish(af, 'pfeil', a.out, 300000, 30.0))
    del fig
    if 'sockel' in parts:
        bm = build_base_mesh()
        log('sockel', len(bm.faces), 'Dreiecke')
        save('sockel', bm)


if __name__ == '__main__':
    main()
