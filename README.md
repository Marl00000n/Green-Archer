# The Green Archer (Columbia-Serial, 1940) – 3D-Druck-Modell

Mehrteiliges, druckoptimiertes STL-Modell des **Green Archer** im Stil des Columbia-Serials von 1940:
Figur im Moment des vollen Auszugs (gespannter Langbogen, aufgelegter Pfeil, Köcher) auf einem Sockel mit Gravur.

![Vorschau](vorschau/Vorschau_Uebersicht.png)

## Dateien

Druckfertig ausgerichtet in `stl/druck/`:

| Datei | Inhalt | Maße (mm) | Dreiecke | Volumen |
|---|---|---|---|---|
| `GreenArcher_01_Figur.stl` | Figur mit Kappe, Feder, Gesichtstuch, Wams, Köcher, Händen, Stiefeln | 194 × 58 × 282 | 1 000 000 | 379 cm³ |
| `GreenArcher_02_Sockel.stl` | Sockel mit Gravur „THE GREEN ARCHER · 1940“, 4 Zapfenbuchsen | 140 × 72 × 30 | 3 076 | 243 cm³ |
| `GreenArcher_03_Bogen_oben.stl` | obere Bogenhälfte, Griffwicklung, Hornnocke, halbe Sehne | 127 × 132 × 9 | 260 010 | 2,7 cm³ |
| `GreenArcher_04_Bogen_unten.stl` | untere Bogenhälfte, dito | 127 × 139 × 9 | 264 596 | 2,8 cm³ |
| `GreenArcher_05_Pfeil.stl` | Pfeil mit Nockkerbe, drei Federn, Bindegarn, Blattspitze | 143 × 7 × 6 | 118 896 | 1,0 cm³ |

`stl/GreenArcher_komplett_zusammengebaut.stl` zeigt alles in Endposition (nur zum Ansehen, reduzierte
Dreieckszahl, 206 × 72 × 402 mm).

Alle fünf Teile sind **wasserdicht und jeweils ein einziger geschlossener Körper**. Die Passungen sind
rechnerisch geprüft: **0 mm³ Überschneidung** zwischen allen Teilepaaren.

## Größe

* Figur: **ca. 28 cm** (Sohle bis Federspitze), mit Sockel ca. 31 cm.
* Zusammengebaut mit Bogenspitzen ca. **40 cm** hoch.
* Für Drucker mit ~250 mm Bauhöhe alle Teile gleichmäßig auf **ca. 75–80 %** skalieren
  (Zapfen und Buchsen skalieren mit, die Passung bleibt erhalten).

## Zusammenbau

1. Figur mit den vier Fußzapfen (Ø 5,6 mm) in den Sockel stecken – Buchsen Ø 6,1 mm, also 0,25 mm Spiel.
2. Die beiden Bogenhälften von oben und unten in die Bohrung der linken Faust schieben
   (Bohrung Ø 6,9 mm, Zapfen Ø 6,4 mm) und verkleben. Die Griffwicklung liegt dann direkt über
   und unter der Faust, die Sehnenenden treffen sich am Nockpunkt neben der Wange.
3. Pfeil mit der Nockkerbe an den Nockpunkt setzen; er liegt hinten in der Rille der Zugfinger
   und vorn auf dem Knöchel der Bogenhand auf.

## Druck-Empfehlung

* **Resin:** 0,03–0,05 mm Schichthöhe. Figur etwa 30° gekippt aufstellen, Stützen automatisch erzeugen.
  Bei dieser Auflösung kommen Nähte, Schnürung und die Lederstruktur sauber heraus.
* **FDM:** 0,1 mm Schichthöhe, Düse 0,25–0,3 mm (mit 0,4 mm gehen die feinsten Nähte verloren),
  Tree-Supports. Bogenhälften und Pfeil liegen flach auf dem Bett.
* **Kleinste Strukturen im Modell:** Sehne Ø 1,8 mm · Pfeilschaft Ø 2,8 mm · Befiederung 0,8 mm dick ·
  Schnürsenkel Ø 1,2–1,4 mm · Steppnähte 0,5–0,6 mm · Bogenarm 2,1–3,8 mm dick.
  Die Sehne ist die empfindlichste Stelle – sie darf notfalls weggeschnitten werden, der Bogen steht auch ohne.
* Für den Look: Kappe, Wams und Tuch grün bemalen, Leder braun, Haut und Sockel abgesetzt.

## Was modelliert ist

* **Hände:** vier gegliederte Finger je Hand mit Knöcheln, Beugefalten und Nägeln. Die linke umfasst
  den Bogengriff, die rechte zieht die Sehne im mediterranen Griff – Zeigefinger über, Mittel- und
  Ringfinger unter dem Pfeil, Sehne in der Endgelenkfalte.
* **Kopf:** Gesicht unter dem anliegenden Tuch – Brauenwulst, Augenöffnungen mit Augapfel und Lidern,
  Nase mit Flügeln, Lippen, Kinn, Wangenknochen, Zugfalten zu den Ohren, Knoten mit zwei Zipfeln im Nacken.
* **Kappe:** Rollrand mit Steppnaht, drei Kappennähte, vorderer Schirm, nach hinten sackende Haube,
  Feder mit Kiel, Fahnenkerben und Wicklung.
* **Kleidung:** Wams mit Besatzkante, Brustschnürung, Schulter- und Seitennähten, kurzen Ärmeln mit
  Saumwulst und Falten; Rock mit elf modellierten Falten und welligem Saum; Gürtel mit Schnalle,
  Löchern, hängendem Riemenende und Steppnähten.
* **Ausrüstung:** Köcher mit sieben Pfeilen, zwei Riemen mit Schnallen, Nieten und Längsnaht;
  Schultergurt mit beidseitiger Steppnaht; Armschiene mit Nähten und Nieten; Gürtelbeutel.
* **Stiefel:** anatomische Schuhform mit Sohle, Absatz und Sohlenprofil, Kreuzschnürung mit zwölf Ösen
  und Schleifenenden, Kappennaht, hintere Längsnaht, Umschlag.
* **Oberfläche:** feines Stoffgewebe auf Rock, Kragen, Ärmeln und Hose, Ledernarbung auf Gürtel, Köcher,
  Stiefeln und Armschiene, Holzmaserung am Bogen, dazu eine gleichmäßige Mikro-Rauheit.

## Was gegenüber deiner Bildvorlage angepasst wurde – und warum

**Film-Optik.** Die auffindbaren Beschreibungen des Serials nennen eine spitze Robin-Hood-Kappe mit Feder,
Strumpfhose, Wams (Jerkin) und ein lose sitzendes Tuch über dem Gesicht
([TCM](https://tcm.com/tcmdb/title/496387/the-green-archer),
[Medleyana-Rezension](https://medleyana.com/2015/06/22/fates-worse-than-death-the-green-archer-1940/)).
Die Bildvorlage zeigt dagegen einen enganliegenden Ganzkörperanzug mit Kapuze und langem Umhang. Umgesetzt
ist deshalb die Serial-Variante; den langen Umhang gibt es nicht, er kommt in keiner Beschreibung vor und
wäre zudem schlecht druckbar.

**Pose.** Haltung, Stand und Blickrichtung folgen der Bildvorlage. Korrigiert wurde der Auszug: in der
Vorlage reicht der Pfeil weit über den Kopf hinaus. Jetzt entspricht die Auszugslänge mit 122 mm
(≈ 780 mm in Originalgröße) einem echten Langbogen, und die Zughand sitzt als Ankerpunkt am Kiefer.

**Druckbarkeit.** Bogen, Pfeil und Sockel sind eigene Teile. Der Bogen ist geteilt, damit er durch die
Faust gesteckt werden kann. Alle Kontaktflächen sind sauber freigeschnitten, die Sehne liegt in einer
Rille der Zugfinger, damit sie rund bleibt.

## Ehrliche Einschränkungen

* Das Modell ist **prozedural erzeugt** (Skript in `build/`), nicht von Hand gesculptet. Anatomie und
  Gesicht sind stilisiert; wer eine Studio-Miniatur erwartet, wird den Unterschied sehen.
* Ich konnte **keine Filmbilder** einsehen, nur Textbeschreibungen. Schnitt der Kappe, Länge des Wamses
  und Stiefelform sind eine plausible Interpretation, kein exakter Nachbau. Ein Screenshot aus dem
  Serial würde helfen, das zu schärfen.
* **Nicht im Slicer oder Drucker getestet.** Geprüft sind per Skript: Wasserdichtheit, Anzahl der
  Teilkörper, Überschneidungsfreiheit und Maße.
* Die ausgestreckten Arme, die Feder, der Köcher und die Bogenspitzen brauchen Stützen.

## Neu erzeugen

```bash
pip install numpy scipy scikit-image trimesh manifold3d fast-simplification shapely mapbox-earcut matplotlib pillow
python3 build/green_archer.py --h 0.25 --out stl/zusammengebaut --faces 1000000   # ca. 6 Minuten
python3 build/export_print.py        # Druckausrichtung, Kontaktschnitte, Passungsprüfung
python3 build/make_previews.py       # Vorschaublatt
```

`--h` ist die Voxelgröße in mm (0.25 = Endqualität, 0.35 = schneller Testlauf).
Pose, Kopfdrehung, Bogenlänge und Auszug stehen als Konstanten am Anfang von `build/green_archer.py`.
