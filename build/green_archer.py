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
HEAD_C = v3(0, -1.0, 252.5)
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


S_L, S_R = v3(27, 0, 219), v3(-27, 0, 219)           # Schultergelenke
E_L, W_L = v3(72, -8, 231), v3(112, -13, 240)         # Bogenarm: Ellbogen, Handgelenk
F_L = v3(120, -13, 240)                                # Faust = Griffmitte des Bogens
E_R, W_R = v3(-62, 3, 238), v3(-26, -12.5, 244.5)      # Zugarm
BOW_Y = -13.0                                          # Ebene des Bogens
ARROW_Y, ARROW_Z = -17.5, 248.6                        # Pfeillage
NOCK = v3(-14.5, -17.0, 248.6)                         # Nockpunkt der Sehne
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
         amp=0.0):
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


# ============================================================== FIGUR
def build_figure(h, quick=False):
    fig = Field(lo=(-88, -40, -12), hi=(143, 52, 298), h=h)
    add, sub = fig.add, fig.sub
    log('Figur: Feld', fig.f.shape, f'{fig.f.size / 1e6:.0f} M Voxel')

    # ------------------------------------------------------------ Beine + Stiefel
    for s in SIDE:
        hip, knee, ank = HIP[s], KNEE[s], ANK[s]
        add(capsule(hip, knee, 15.0, 11.2), 6)
        add(capsule(knee, ank + v3(0, 0, 3), 11.0, 7.0), 5)
        Rth = frame_from_axis(knee - hip)
        add(ellipsoid(hip + (knee - hip) * 0.42 + v3(2.0 * s, -4.0, 0), (9.0, 8.5, 21.0), Rth), 5)  # Quadrizeps
        add(ellipsoid(hip + (knee - hip) * 0.40 + v3(-3.0 * s, 4.0, 0), (8.0, 7.5, 19.0), Rth), 5)  # Beuger
        add(ellipsoid(knee + v3(0, -4.4, -1.0), (6.2, 4.6, 6.0)), 3)                              # Kniescheibe

        d = FOOT_D[s]
        z = v3(0, 0, 1)
        u = unit(np.cross(d, z))
        Rf = np.array([u, d, z])
        add(ellipsoid(ank + d * -2.0 + v3(0, 0, -0.5), (7.7, 9.6, 10.4), Rf), 4)   # Ferse/Knoechel
        add(ellipsoid(ank + d * 9.5 + v3(0, 0, -4.0), (7.4, 11.2, 6.9), Rf), 4)    # Spann
        add(ellipsoid(ank + d * 19.5 + v3(0, 0, -5.9), (6.6, 8.4, 5.4), Rf), 3)    # Kappe
        add(rbox(ank + d * 7.0 + v3(0, 0, -9.9), (8.1, 18.8, 1.1), Rf, 0.9), 0.8)  # Sohle
        add(rbox(ank + d * -4.5 + v3(0, 0, -8.9), (7.7, 4.4, 2.1), Rf, 0.6), 0.8)  # Absatz
        # Schaft + Umschlag
        sd = unit(knee - ank)
        top = ank + sd * ((57 - ank[2]) / sd[2])
        add(capsule(ank + v3(0, 0, 4), top, 7.4, 9.0), 4)
        add(torus(top - sd * 3.2, sd, 9.3, 2.5), 1.2)                 # Umschlag
        add(torus(top - sd * 7.0, sd, 9.0, 0.9), 0.6)                 # Naht unter Umschlag
        # Kreuzschnuerung am Schaft
        for i in range(5):
            t = 0.12 + 0.15 * i
            c = ank + (top - ank) * t
            rr = 7.4 + (9.0 - 7.4) * t + 0.15
            tilt = (-1) ** i * 0.22
            ax = unit(sd + u * tilt)
            add(torus(c, ax, rr, 0.8), 0.5)
        # Sohlennaht / Zehenkappe
    log('Beine fertig')

    # --------------------------------------------------------------- Torso
    zz = [112, 118, 135, 150, 165, 180, 195, 208, 218, 224]
    aa = [23.0, 23.5, 23.0, 20.5, 22.0, 25.5, 27.5, 27.0, 23.0, 17.0]
    bb = [15.0, 15.5, 15.5, 14.2, 15.0, 16.5, 17.5, 16.5, 14.5, 12.0]
    fa, fb = PchipInterpolator(zz, aa), PchipInterpolator(zz, bb)
    add(loft(112, 224, lambda z: fa(z), lambda z: fb(z), cy=lambda z: 1.0 + 0 * z, rnd=2.0), 4)
    for s in SIDE:                                                  # Brustmuskeln
        add(ellipsoid(v3(11.0 * s, -10.5, 194), (9.5, 5.0, 6.8)), 6)
    add(ellipsoid(v3(0, 3, 214), (17, 9, 7)), 5)                    # Trapez
    add(capsule(S_R, S_L, 11.0, 11.0), 6)                           # Schultern
    log('Torso fertig')

    # ----------------------------------------------- Wams-Rock, Guertel, Falten
    def skirt_ab(z):
        t = np.clip((z - 106) / 44.0, 0, 1)
        return 28.2 + (21.6 - 28.2) * t ** 1.3, 20.2 + (14.8 - 20.2) * t ** 1.3

    pleat = lambda phi, z: 0.75 * np.sin(13 * phi + 0.05 * z) * (1 - np.clip((z - 104) / 46.0, 0, 1)) ** 0.8
    add(loft(106, 152, lambda z: skirt_ab(z)[0], lambda z: skirt_ab(z)[1], cy=lambda z: 1.0 + 0 * z,
             mod=pleat, zmod=lambda phi: 1.6 * np.sin(7 * phi), amp=1.5, rnd=1.0), 2)
    # Seitenschlitze + Mittelschlitz am Saum
    for x0 in (-1, 1):
        sub(rbox(v3(x0 * 28.6, 1.0, 110.5), (1.4, 7.0, 7.5), None, 0.5), 0.4)
    sub(rbox(v3(0, -21.2, 109.0), (0.8, 2.5, 6.0), None, 0.4), 0.3)
    # Blousing ueber dem Guertel
    add(loft(150, 160, lambda z: 22.8 - 0.25 * (z - 150), lambda z: 16.4 - 0.2 * (z - 150),
             cy=lambda z: 1.0 + 0 * z, mod=lambda phi, z: 0.6 * np.sin(11 * phi), rnd=2.0), 3)
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
    # Zugarm (rechts, -x)
    add(ellipsoid(S_R + v3(-1.0, 0, 0.5), (13.0, 12.0, 12.5)), 6)
    add(capsule(S_R, E_R, 10.2, 8.3, ripple=ripple_s), 4)
    add(capsule(E_R, W_R, 8.2, 5.4, ripple=(0.16, 1.3, 2.8, 2.0)), 4)
    add(torus(W_R + unit(E_R - W_R) * 1.5, W_R - E_R, 5.5, 1.1), 0.6)       # Manschette
    log('Arme fertig')

    # ---------------------------------------------------------- Linke Faust (Bogenhand)
    F = F_L
    add(rbox(F, (6.6, 6.2, 7.6), None, 3.0), 1.5)
    add(ellipsoid(F + v3(-4.0, 0.5, 0), (4.5, 6.0, 6.5)), 2)                 # Handballen
    for zi in (-5.6, -1.9, 1.9, 5.6):                                         # Knöchel / Finger
        add(capsule(F + v3(-2.0, -5.1, zi), F + v3(4.0, -5.3, zi), 1.85), 0.6)
    for zi in (-3.8, 0.0, 3.8):                                               # Fingerfugen
        sub(capsule(F + v3(-1.5, -7.3, zi), F + v3(6.8, -6.0, zi), 0.35), 0.2)
    add(capsule(F + v3(-4.0, 5.0, 4.5), F + v3(3.5, 4.8, 3.2), 2.1, 1.8), 1.0)  # Daumen
    sub(cylinder(F + v3(0, 0, -9.0), F + v3(0, 0, 9.0), 2.65))                # Bohrung fuer Bogengriff
    # ---------------------------------------------------------- Rechte Hand (Zughand)
    HR = v3(-21.5, -13.2, 244.0)
    Rpalm = frame_from_axis(W_R - E_R)
    add(ellipsoid(HR, (3.8, 5.6, 6.0), Rpalm), 2)
    fz = (251.4, 244.6, 241.2)
    for zi in fz:
        k0 = v3(-23.5, -12.6, zi + (HR[2] - 245.0) * 0.1)
        k1 = v3(-17.5, -14.6, zi)
        k2 = v3(-14.2, -16.4, zi)
        add(capsule(k0, k1, 1.75, 1.6), 0.8)
        add(capsule(k1, k2, 1.6, 1.35), 0.6)
    add(capsule(v3(-24.5, -9.5, 241.0), v3(-18.5, -9.0, 237.6), 2.0, 1.6), 1.0)  # Daumen
    log('Haende fertig')

    # --------------------------------------------------------- Kopf, Tuch, Muetze
    rh = HEAD_R
    add(ellipsoid(HEAD_C, rh, R_H), 2)
    # Tuch (Gesichtsschleier + Kapuze): Kopfhuelle etwas groesser
    add(ellipsoid(HEAD_C, (rh[0] + 0.9, rh[1] + 0.9, rh[2] + 0.9), R_H), 1.5)
    add(ellipsoid(hsurf(0, -55, -0.6), (4.6, 3.8, 3.2), R_H), 2.5)                       # Kinn
    # Gesichtsform unter dem Tuch (alle Punkte liegen exakt auf der Tuchoberflaeche)
    add(capsule(hsurf(0, 14, 0.2), hsurf(0, -6, 1.6), 1.6, 2.6), 2.0)                # Nasenruecken
    add(ellipsoid(hsurf(0, -9.5, 2.0), (2.4, 2.5, 1.9), R_H), 1.0)                  # Nasenspitze
    for sgn in (-1, 1):
        add(capsule(hsurf(sgn * 6, -10, 0.8), hsurf(sgn * 15, -12, 0.6), 1.3, 1.1), 1.2)   # Nasenfluegel
        add(ellipsoid(hsurf(sgn * 24, -4, 0.0), (4.2, 3.6, 3.0), np.array(
            [math.cos(math.radians(sgn * 24)) * U_H - math.sin(math.radians(sgn * 24)) * V_H,
             math.sin(math.radians(sgn * 24)) * U_H + math.cos(math.radians(sgn * 24)) * V_H, W_H])), 2.5)  # Wangenknochen
    # Augenoeffnungen mit Augapfel und Lidwulst
    for sgn in (-1, 1):
        ph = math.radians(sgn * 25)
        vo = math.sin(ph) * U_H + math.cos(ph) * V_H
        uo = math.cos(ph) * U_H - math.sin(ph) * V_H
        Re = np.array([uo, vo, W_H])
        c = hsurf(sgn * 25, 12, 0.0)
        sub(ellipsoid(c + vo * 0.2, (3.5, 3.2, 1.9), Re), 0.5)
        add(ellipsoid(c - vo * 2.3, (2.0, 1.8, 1.35), Re), 0.3)
        add(capsule(hsurf(sgn * 14, 22, 0.1), hsurf(sgn * 40, 20, 0.1), 0.85, 0.7), 0.6)   # Stirn-/Lidwulst
    # Falten (Zugfalten des Tuchs) und Mund
    for sgn in (-1, 1):
        sub(capsule(hsurf(sgn * 9, -7, 0.0), hsurf(sgn * 40, -17, 0.0), 0.42, 0.5), 0.4)
        sub(capsule(hsurf(sgn * 8, -16, 0.0), hsurf(sgn * 44, -31, 0.0), 0.38, 0.5), 0.4)
        sub(capsule(hsurf(sgn * 14, -27, 0.0), hsurf(sgn * 52, -40, 0.0), 0.34, 0.45), 0.4)
    sub(capsule(hsurf(-15, -27, 0.0), hsurf(15, -27, 0.0), 0.4), 0.4)
    # Knoten hinten + Zipfel
    add(ellipsoid(hw(0, -14.5, -2.0), (3.3, 2.8, 3.0), R_H), 1.0)
    add(capsule(hw(0.5, -15.0, -3.0), hw(2.5, -17.0, -16.0), 1.5, 0.7), 0.8)
    add(capsule(hw(-0.5, -15.0, -3.0), hw(-3.5, -16.4, -13.0), 1.4, 0.6), 0.8)

    # Kapuzen-Kragen (Cowl): faellt vom Kopf auf die Schultern
    def cowl_ab(z):
        t = np.clip((z - 214.0) / 30.0, 0, 1)
        return 12.8 + (31.0 - 12.8) * (1 - t) ** 2.3, 12.8 + (18.5 - 12.8) * (1 - t) ** 2.3

    cmod = lambda phi, z: (0.95 * np.sin(11 * phi + 0.12 * z)
                           * np.clip(1 - (z - 214.0) / 30.0, 0, 1) ** 0.6)
    add(loft(214, 246, lambda z: cowl_ab(z)[0], lambda z: cowl_ab(z)[1],
             cy=lambda z: 2.0 + 0 * z, mod=cmod, zmod=lambda phi: 1.4 * np.sin(9 * phi), amp=1.5, rnd=1.2), 3)
    log('Kopf fertig')

    # ------------------------------------------------- Muetze (Robin-Hood-Kappe)
    cap_top = ellipsoid(hw(0, -1.0, 9.6), (13.4, 15.6, 12.0), R_H)
    add(cap_top, 2)
    fn_b, bmn, bmx = head_shell(1.4, 3.2, 6.2, 11.0)      # umlaufender Rollrand
    add(custom(fn_b, bmn, bmx), 0.8)
    # Naehte der Kappe
    for ang in (-50, 0, 50):
        a = math.radians(ang)
        p0 = hw(math.sin(a) * 14.0, -1.0 + math.cos(a) * 15.6, 11.0)
        p1 = hw(math.sin(a) * 4.0, -1.0 + math.cos(a) * 4.0, 21.5)
        sub(capsule(p0, p1, 0.45), 0.3)
    # vordere Spitze (Schirm)
    pk = [(0, 12.5, 8.4, 7.0, 2.0), (0, 17.0, 7.6, 6.2, 1.7), (0, 21.5, 6.6, 4.6, 1.4), (0, 26.0, 5.6, 2.6, 1.1), (0, 29.5, 4.9, 1.2, 0.8)]
    Rpk = np.array([U_H, V_H, W_H])
    for (u, v, w, ru, rw) in pk:
        add(ellipsoid(hw(u, v, w), (ru, 3.0, rw), Rpk), 1.5)
    # weiche Kappen-Kuppe, leicht nach hinten gekippt
    add(ellipsoid(hw(0, -4.0, 17.5), (8.0, 9.0, 6.5), R_H), 4.0)

    # Feder
    fpts = [hw(11.0, 7.0, 9.5), hw(14.8, -1.0, 12.5), hw(17.5, -10.0, 13.0),
            hw(19.5, -19.0, 10.0), hw(20.5, -27.0, 5.0), hw(20.0, -33.0, -2.5)]
    path = catmull([tuple(p) for p in fpts], 14)
    n = len(path)
    for i in range(n - 1):
        t = i / (n - 1)
        tan = unit(path[i + 1] - path[i])
        side = U_H * (1.0 if True else -1)
        wdir = unit(np.cross(tan, side))
        Rf = np.array([tan, wdir, unit(np.cross(tan, wdir))])
        width = 5.2 * math.sin(math.pi * min(1, t * 1.05 + 0.05) ** 0.8) + 0.5
        c = (path[i] + path[i + 1]) / 2
        add(ellipsoid(c, (np.linalg.norm(path[i + 1] - path[i]) * 0.75, width, 0.95), Rf), 0.6)
    add(capsule(path[0], path[-1] * 0 + path[n // 2], 1.0, 0.7), 0.5)
    for i in range(0, n - 4, 2):                                         # Kiel
        add(capsule(path[i], path[i + 2], 0.8, 0.7), 0.3)
    for i in range(3, n - 3, 2):                                         # Federfahnen-Kerben
        tan = unit(path[i + 1] - path[i - 1])
        wdir = unit(np.cross(tan, U_H))
        for sg in (-1, 1):
            t = i / (n - 1)
            wd = 5.2 * math.sin(math.pi * t ** 0.8) + 0.5
            p0 = path[i] + wdir * sg * 0.6
            p1 = path[i] + wdir * sg * wd * 1.05 - tan * 1.8
            sub(capsule(p0 + U_H * 0.9, p1 + U_H * 0.9, 0.28), 0.1)
            sub(capsule(p0 - U_H * 0.9, p1 - U_H * 0.9, 0.28), 0.1)
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
    for i in range(6):
        zc = 205 - i * 8.2
        ysf = fig.surface_y(0.0, zc, -40, 10, 0.1)
        if ysf is None:
            continue
        add(capsule(v3(-3.2, ysf - 0.2, zc + 1.8), v3(3.2, ysf - 0.2, zc - 1.8), 0.75), 0.3)
        add(capsule(v3(3.2, ysf - 0.2, zc + 1.8), v3(-3.2, ysf - 0.2, zc - 1.8), 0.75), 0.3)
    ys1 = fig.surface_y(0.0, 210, -40, 10, 0.1)
    if ys1 is not None:
        sub(capsule(v3(0, ys1 - 0.3, 211), v3(0, ys1 - 0.3, 160), 0.5), 0.2)

    # ----------------------------------------------------------- Boden kappen + Zapfen
    sub(rbox(v3(0, 0, -30.0), (200, 200, 30.0), None, 0.0), 0.0)
    for (px, py) in peg_positions():
        add(cylinder(v3(px, py, -PEG_LEN), v3(px, py, 0.6), PEG_R, 0.6), 0.5)
    log('Figur: Feld fertig')
    return fig


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
    z0 = BOW_ZG + sign * 0.35
    z1 = BOW_ZG + sign * (FIST_HH + 0.4)
    f.add(cylinder(v3(BOW_XG, BOW_Y, z0), v3(BOW_XG, BOW_Y, z1), 2.35, 0.5), 0.3)
    # Griffwicklung ueber der Faust
    z2 = BOW_ZG + sign * (FIST_HH + 11.0)
    f.add(cylinder(v3(BOW_XG, BOW_Y, BOW_ZG + sign * (FIST_HH + 0.4)), v3(BOW_XG, BOW_Y, z2), 3.2, 0.6), 0.3)
    for k in range(4):
        zc = BOW_ZG + sign * (FIST_HH + 2.2 + 2.6 * k)
        f.add(torus(v3(BOW_XG, BOW_Y, zc), v3(0, 0, 1), 3.2, 0.55), 0.3)
    # Wurfarm
    f.add(limb_shape(sign), 1.2)
    # Nocke an der Spitze
    tip = v3(bow_x(BOW_HALF), BOW_Y, BOW_ZG + sign * BOW_HALF)
    f.add(ellipsoid(tip, (2.6, 2.7, 3.4)), 0.8)
    # Sehnenhaelfte bis zum Nockpunkt
    f.add(capsule(tip + v3(-1.8, 0, -sign * 0.8), NOCK, 0.95, 0.95), 0.4)
    if sign > 0:
        f.add(ellipsoid(NOCK, (1.5, 1.5, 1.5)), 0.3)
    return f


# ================================================================ PFEIL
def build_arrow(h):
    x0, x1 = NOCK[0] - 3.5, 178.5
    lo = (x0 - 2, ARROW_Y - 10, ARROW_Z - 10)
    hi = (x1 + 2, ARROW_Y + 10, ARROW_Z + 10)
    f = Field(lo, hi, h)
    c0 = v3(NOCK[0] - 2.5, ARROW_Y, ARROW_Z)
    f.add(capsule(c0, v3(160.0, ARROW_Y, ARROW_Z), 1.35, 1.35), 0.2)
    f.add(ellipsoid(c0, (2.2, 1.9, 1.9)), 0.4)                       # Nock-Ende
    f.sub(rbox(c0 + v3(-1.4, 0, 0), (1.2, 0.35, 3.0), None, 0.1), 0.1)  # Kerbe fuer die Sehne
    # Befiederung: drei Federn (eine nach unten, zwei schraeg nach oben)
    for ang_deg in (0, 120, 240):
        ang = math.radians(ang_deg)
        nrm = v3(0, math.cos(ang), math.sin(ang))
        tng = v3(1, 0, 0)
        bin_ = np.cross(tng, nrm)
        R = np.array([tng, nrm, bin_])
        for i in range(34):
            t = i / 33
            xc = -6.0 + 23.0 * t
            hgt = 4.2 * (0.4 + 0.6 * math.sin(math.pi * min(1, 0.2 + 0.8 * t))) * (1 - 0.65 * t ** 3)
            cc = v3(xc, ARROW_Y, ARROW_Z) + nrm * (hgt / 2 + 0.5)
            f.add(rbox(cc, (1.5, hgt / 2 + 0.5, 0.55), R, 0.4), 0.25)
    # Spitze: Blattspitze
    f.add(capsule(v3(158, ARROW_Y, ARROW_Z), v3(163, ARROW_Y, ARROW_Z), 1.5, 2.0), 0.3)
    pts = [(163, 0.0), (166.5, 3.0), (172, 3.4), (178.2, 0.0)]
    for (xa, wa), (xb, wb) in zip(pts[:-1], pts[1:]):
        f.add(rbox(v3((xa + xb) / 2, ARROW_Y, ARROW_Z), ((xb - xa) / 2 + 0.3, max(wa, wb) / 2 + 0.2, 0.9),
                   None, 0.3), 0.2)
    f.add(capsule(v3(172, ARROW_Y, ARROW_Z), v3(178.4, ARROW_Y, ARROW_Z), 1.2, 0.25), 0.2)
    for xr in (27.0, 29.2):                                          # Zierringe
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
    segs['Pfeilschaft'] = (v3(NOCK[0] - 2.5, ARROW_Y, ARROW_Z), v3(160, ARROW_Y, ARROW_Z), 1.35)
    out = {}
    for k, (a, b, r) in segs.items():
        t = np.linspace(0, 1, 400)[:, None]
        pts = a + (b - a) * t
        out[k] = float(fig.sample(pts).min() - r)
        log(f'  Abstand {k}: {out[k]:+.2f} mm zur Figur (negativ = Ueberschneidung)')
    return out


def finish(f, name, outdir, target_faces=None, min_vol=1.0):
    m = f.mesh()
    log(name, 'Marching Cubes:', len(m.faces), 'Dreiecke')
    m = clean_mesh(m, min_vol)
    comps = m.split(only_watertight=False)
    log(name, 'roh wasserdicht:', m.is_watertight, ' Komponenten:', len(comps))
    if len(comps) > 1:
        for c in comps:
            log('   Komponente: Volumen %.1f mm3, Dreiecke %d, bounds %s' % (abs(c.volume), len(c.faces), np.round(c.bounds, 0).tolist()))
    if target_faces and len(m.faces) > target_faces:
        import fast_simplification as fs
        v, fcs = fs.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                             target_reduction=1 - target_faces / len(m.faces), agg=4)
        m2 = trimesh.Trimesh(v, fcs, process=True)
        m2.fix_normals()
        log(name, 'vereinfacht auf', len(m2.faces), 'Dreiecke, wasserdicht:', m2.is_watertight)
        if m2.is_watertight or not m.is_watertight:
            m = m2
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_winding(m)
        log(name, 'nach Reparatur wasserdicht:', m.is_watertight)
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
    hb = min(a.h, 0.25)
    meshes = {}

    def save(name, m):
        meshes[name] = m
        m.export(os.path.join(a.out, f'{name}_assembled.stl'))
        log(name, 'watertight=', m.is_watertight, 'Volumen=%.0f mm3' % m.volume, 'bounds', np.round(m.bounds, 1).tolist())

    fig = build_figure(a.h) if ('figur' in parts or 'bogen' in parts or 'pfeil' in parts) else None
    if fig is not None:
        clearance_report(fig)
    if 'figur' in parts:
        save('figur', finish(fig, 'figur', a.out, a.faces, 20.0))
    if 'bogen' in parts:
        for sg, nm in ((1, 'bogen_oben'), (-1, 'bogen_unten')):
            bf = build_bow_half(sg, hb)
            carve(bf, fig)
            save(nm, finish(bf, nm, a.out, 300000, 0.5))
    if 'pfeil' in parts:
        af = build_arrow(hb)
        carve(af, fig)
        save('pfeil', finish(af, 'pfeil', a.out, 300000, 0.2))
    del fig
    if 'sockel' in parts:
        bm = build_base_mesh()
        log('sockel', len(bm.faces), 'Dreiecke')
        save('sockel', bm)


if __name__ == '__main__':
    main()
