#!/usr/bin/env python3
"""Erzeugt das Vorschaublatt (Ansichten, Details, Druckteile) aus den fertigen STL-Dateien."""
import os, sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
import matplotlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preview import render, VIEWS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SRC = os.path.join(ROOT, 'stl', 'zusammengebaut')
DRK = os.path.join(ROOT, 'stl', 'druck')
OUT = os.path.join(ROOT, 'vorschau')
os.makedirs(OUT, exist_ok=True)
FONT = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSans.ttf')
FONTB = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')
BG = (26, 26, 30)
STONE, BASE, WOOD = (206, 204, 197), (180, 178, 172), (214, 196, 160)

def L(p):
    return trimesh.load(p)

fig = L(f'{SRC}/figur_assembled.stl')
sock = L(f'{SRC}/sockel_assembled.stl'); sock.apply_translation([0, 0, -0.3])
bo, bu, pf = L(f'{SRC}/bogen_oben_assembled.stl'), L(f'{SRC}/bogen_unten_assembled.stl'), L(f'{SRC}/pfeil_assembled.stl')
ALL = [(fig, STONE), (sock, BASE), (bo, WOOD), (bu, WOOD), (pf, WOOD)]

def fit(im, w, h, pad=6):
    im = im.copy(); im.thumbnail((w - 2 * pad, h - 2 * pad), Image.LANCZOS)
    c = Image.new('RGB', (w, h), BG); c.paste(im, ((w - im.width) // 2, (h - im.height) // 2)); return c

def label(c, text, size=20):
    d = ImageDraw.Draw(c); f = ImageFont.truetype(FONT, size)
    return d.text((10, c.height - size - 8), text, fill=(235, 235, 235), font=f)

ref_bounds = None
sx = []
cells1 = []
V = np.concatenate([m.vertices for m, _ in ALL])
for key, name in (('front', 'Vorderansicht'), ('right', 'Rechte Seitenansicht'), ('back', 'Rückansicht'),
                  ('left', 'Linke Seitenansicht'), ('top', 'Draufsicht')):
    im = render(ALL, VIEWS[key], s=2.0)
    cells1.append((im, name))
    print('Ansicht', name, flush=True)

details = [
    ('Detail: Maske / Kopf', 'q34', (-38, 34, -30, 22, 226, 292), 8),
    ('Detail: Rechte Hand (Zug)', 'q34', (-50, -2, -34, 12, 228, 262), 14),
    ('Detail: Linke Hand (Halt)', 'q34', (100, 138, -30, 12, 222, 258), 14),
    ('Detail: Oberkörper / Kleidung', 'q34', (-36, 36, -30, 24, 140, 226), 7),
    ('Detail: Köcher', 'back', (-52, 2, 0, 40, 100, 232), 8),
    ('Detail: Stiefel / Sockel', 'q34', (20, 78, -36, 30, -31, 68), 8),
]
cells2 = []
for name, v, c, s in details:
    parts = [(m, col) for m, col in ALL]
    im = render(parts, VIEWS[v], s=s, crop=c)
    cells2.append((im, name)); print('Detail', name, flush=True)

pf_parts = []
d = {n: L(f'{DRK}/GreenArcher_{n}.stl') for n in ('01_Figur', '02_Sockel', '03_Bogen_oben', '04_Bogen_unten', '05_Pfeil')}
cells3 = [
    (render([(d['01_Figur'], STONE)], VIEWS['q34'], s=1.6), 'Figur'),
    (render([(d['03_Bogen_oben'], WOOD), (trimesh.Trimesh(d['04_Bogen_unten'].vertices + [d['03_Bogen_oben'].extents[0] + 12, 0, 0], d['04_Bogen_unten'].faces), WOOD)], VIEWS['top'], s=1.6), 'Bogenhälften (flach, wie gedruckt)'),
    (render([(d['05_Pfeil'], WOOD)], VIEWS['top'], s=2.2), 'Pfeil'),
    (render([(d['02_Sockel'], BASE)], VIEWS['q34'], s=3.0), 'Sockel'),
]

W = 2400
def row(cells, ch, title=None):
    cw = W // len(cells)
    r = Image.new('RGB', (W, ch), BG)
    for i, (im, nm) in enumerate(cells):
        c = fit(im, cw, ch); label(c, nm); r.paste(c, (i * cw, 0))
    return r
r1 = row(cells1, 760); r2 = row(cells2, 420); r3 = row(cells3, 520)
title_h = 90
sheet = Image.new('RGB', (W, title_h + r1.height + r2.height + r3.height), BG)
dr = ImageDraw.Draw(sheet)
dr.text((30, 18), 'THE GREEN ARCHER (Columbia-Serial, 1940) – 3D-Druck-Modell', fill=(240, 240, 240), font=ImageFont.truetype(FONTB, 40))
dr.text((30, 62), 'Figur ca. 28 cm (ohne Sockel) · mehrteilig: Figur, Sockel, 2 Bogenhälften, Pfeil · Vorschau-Renderings der STL-Dateien', fill=(180, 180, 185), font=ImageFont.truetype(FONT, 20))
y = title_h
for r in (r1, r2, r3):
    sheet.paste(r, (0, y)); y += r.height
sheet.save(f'{OUT}/Vorschau_Uebersicht.png')
print('fertig', sheet.size)
