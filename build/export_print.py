#!/usr/bin/env python3
"""Erzeugt aus den zusammengebauten Teilen die druckfertigen STL-Dateien + Passungs-Pruefung."""
import os, sys
import numpy as np
import trimesh

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stl')
SRC = os.path.join(ROOT, 'zusammengebaut')
DST = os.path.join(ROOT, 'druck')
os.makedirs(DST, exist_ok=True)
def L(n):
    """STL laden und als geschlossenes Volumen reparieren (float32-Rundung der STL-Datei)."""
    m = trimesh.load(os.path.join(SRC, f'{n}_assembled.stl'))
    for dig in (5, 4, 3):
        if m.is_volume:
            break
        m.merge_vertices(digits_vertex=dig)
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_winding(m)
        trimesh.repair.fix_normals(m)
    print(f'  {n}: watertight={m.is_watertight} Volumen={m.volume/1000:.2f} cm3 Dreiecke={len(m.faces)}')
    return m
M = {n: L(n) for n in ('figur', 'sockel', 'bogen_oben', 'bogen_unten', 'pfeil')}

# Sockel in der zusammengebauten Lage 0.3 mm tiefer (Fertigungstoleranz der Sohlen)
M['sockel'].apply_translation([0, 0, -0.3])
# saubere Kontaktflaechen: Sehnenenden dort kuerzen, wo sie Pfeil / anderen Bogenteil beruehren
def diff(a_, b_):
    r = trimesh.boolean.difference([a_, b_], engine='manifold')
    r.merge_vertices()
    return r
M['bogen_unten'] = diff(M['bogen_unten'], M['bogen_oben'])
M['bogen_oben'] = diff(M['bogen_oben'], M['pfeil'])
M['bogen_unten'] = diff(M['bogen_unten'], M['pfeil'])
for k in ('bogen_oben', 'bogen_unten'):
    n = len(M[k].split(only_watertight=False))
    print(f'{k}: {n} Komponente(n) nach Kontakt-Schnitt')

# ------------------------------------------------------------ Passungs-Pruefung
print('Passungs-Pruefung (Ueberschneidungsvolumen, soll ~0 sein):')
pairs = [('figur', 'sockel'), ('figur', 'bogen_oben'), ('figur', 'bogen_unten'), ('figur', 'pfeil'),
         ('bogen_oben', 'bogen_unten'), ('bogen_oben', 'pfeil'), ('bogen_unten', 'pfeil')]
for a, b in pairs:
    try:
        inter = trimesh.boolean.intersection([M[a], M[b]], engine='manifold')
        v = abs(inter.volume) if len(inter.faces) else 0.0
    except Exception as e:
        v = float('nan')
    print(f'  {a:11s} x {b:11s}: {v:8.2f} mm3')

# ------------------------------------------------------------ Druck-Ausrichtung
def put_on_bed(m, center_xy=True):
    m = m.copy()
    b = m.bounds
    t = [-(b[0, 0] + b[1, 0]) / 2 if center_xy else 0, -(b[0, 1] + b[1, 1]) / 2 if center_xy else 0, -b[0, 2]]
    m.apply_translation(t)
    return m

out = {}
out['GreenArcher_01_Figur'] = put_on_bed(M['figur'])
out['GreenArcher_02_Sockel'] = put_on_bed(trimesh.Trimesh(M['sockel'].vertices + [0, 0, 0.3], M['sockel'].faces))
Rx90 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])      # (x,y,z)->(x,-z,y): Bogenebene liegt flach
for n, nm in (('bogen_oben', 'GreenArcher_03_Bogen_oben'), ('bogen_unten', 'GreenArcher_04_Bogen_unten')):
    m = M[n].copy(); m.apply_transform(Rx90); out[nm] = put_on_bed(m)
m = M['pfeil'].copy(); m.apply_transform(Rx90); out['GreenArcher_05_Pfeil'] = put_on_bed(m)

for k, m in out.items():
    m.export(os.path.join(DST, k + '.stl'))
    e = m.extents
    print(f'{k:28s} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} mm  watertight={m.is_watertight}  Dreiecke={len(m.faces)}  Volumen={m.volume/1000:.1f} cm3')

# komplette Szene (Vorschau / Skalierung / Anprobe)
import fast_simplification as fs


def light(m, tgt):
    if len(m.faces) <= tgt:
        return m
    v, f = fs.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                       target_reduction=1 - tgt / len(m.faces), agg=5)
    return trimesh.Trimesh(v, f, process=True)


scene = trimesh.util.concatenate([M['sockel'], light(M['figur'], 260000),
                                  light(M['bogen_oben'], 40000), light(M['bogen_unten'], 40000),
                                  light(M['pfeil'], 25000)])
scene.export(os.path.join(ROOT, 'GreenArcher_komplett_zusammengebaut.stl'))
b = scene.bounds
print('Komplett zusammengebaut: %.0f x %.0f x %.0f mm' % tuple(b[1] - b[0]))
