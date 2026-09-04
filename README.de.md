# Sternhaufen finden — Anleitung

*English version: [README.md](README.md)*

Dieses Projekt gehört zu den [Physik-Projekt-Tagen](https://ppt.physik.rwth-aachen.de/) an der
RWTH Aachen. Es geht darum, mit *Python* und echten Messdaten des Weltraumteleskops *Gaia*
nach Sternhaufen zu suchen.

Die meisten Sterne ziehen allein durch die Galaxie. Manche sind aber gemeinsam unterwegs: Sie
sind zusammen aus derselben Gaswolke entstanden und bewegen sich bis heute miteinander. Mit
diesem Code suchst du nach so einer Gruppe — zuerst mit eigenen Augen in dem, was das
Teleskop tatsächlich gemessen hat, danach mit denselben Algorithmen, die auch in der
professionellen Astronomie eingesetzt werden. Anschließend
vergleichst du, was die verschiedenen Herangehensweisen finden.

Es geht nicht darum, am Ende die eine perfekte Liste von Haufenmitgliedern zu haben. Der Sinn
der Übung ist, dieselben Sterne auf mehrere Arten zu untersuchen, zu sehen, worin jede Methode
gut ist, und dabei zu verstehen, wie so eine Suche tatsächlich abläuft.

> **Getestet wurde dieser Code mit den Plejaden und Praesepe** — welche Schritte dabei
> funktioniert haben und welche nicht, steht unten unter „Woran das hier getestet wurde".

> **Die Daten sind nicht enthalten.** Du lädst dir deine eigene Sterntabelle als FITS-Datei aus
> dem Gaia-Archiv herunter und zeigst dem Code, wo sie liegt.

## Vorbereitung

Das Projekt benutzt [uv](https://docs.astral.sh/uv/) und installiert damit alles, was du
brauchst:

```bash
uv sync
uv run jupyter lab
```

Mehr ist nicht nötig. `uv sync` installiert Python und alle Pakete, `jupyter lab` öffnet die
Notebook-Umgebung in deinem Browser.

## Die Daten besorgen

Lege deine Gaia-Datei in den Ordner `data/`. Ein einfaches `SELECT *` funktioniert, liefert dir
aber 152 Spalten, an denen du vorbeisehen musst, und eine rund zehnmal so große Datei wie nötig.
Diese acht Spalten benutzt der Code tatsächlich:

| Spalte | Wofür sie gebraucht wird |
| --- | --- |
| `ra`, `dec` | Ort am Himmel — zum Auswählen von Hand und zum Clustern |
| `parallax` | Die Entfernung, über `add_distance` |
| `pmra`, `pmdec` | Eigenbewegung — zum Auswählen von Hand und zum Clustern |
| `radial_velocity` | Die dritte Geschwindigkeitskomponente, ohne die es kein `U, V, W` gibt |
| `phot_g_mean_mag` | Helligkeit, für die Altersbestimmung |
| `bp_rp` | Farbe, für die Altersbestimmung |

Ein Kreis um einen Haufen, der genau diese Spalten anfordert, sieht im Abfragefeld des Archivs
so aus:

```sql
SELECT ra, dec, parallax, pmra, pmdec, radial_velocity, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 56.60, 24.11, 4))
  AND parallax BETWEEN 3 AND 12
```

Für die Form mit `CONTAINS(...)` ist der räumliche Index des Archivs gebaut; derselbe Kreis als
`DISTANCE(...) < 4` geschrieben kann deutlich langsamer sein. Die Parallaxen-Grenzen halten die
Datei klein und entfernen nebenbei jede unbrauchbare Parallaxe — in deiner Tabelle fehlt dann
keine einzige Entfernung.

`source_id` kannst du ergänzen, wenn du einzelne Sterne wiederfinden willst; der Code braucht
die Spalte nicht.

## Die Notebooks

Der eigentliche Kurs besteht aus drei Notebooks im Ordner `notebooks/`. Arbeite sie der Reihe
nach durch — jedes übergibt sein Ergebnis an das nächste:

| Notebook | Was du machst |
| --- | --- |
| `01_auswahl_von_hand.ipynb` | Den Haufen selbst in den Gaia-Messwerten finden: über die Entfernung, den Ort am Himmel und die Eigenbewegung auswählen, alles kombinieren, in echte Positionen umrechnen und zum Schluss in 3D ansehen |
| `02_clustering_algorithmen.ipynb` | HDBSCAN, DBSCAN und ein Gaussian Mixture Model dieselbe Aufgabe in der Position, in der Geschwindigkeit und im vollen Phasenraum lösen lassen, ihre Regler ausprobieren und mit deiner eigenen Auswahl vergleichen |
| `03_alter_des_haufens.ipynb` | *Optional.* Das Alter des Haufens bestimmen, indem du Modell-Isochronen über sein Farben-Helligkeits-Diagramm legst. Von beiden Notebooks oben aus erreichbar |

Der Rest dieser Seite ist derselbe Ablauf noch einmal als reiner Code — praktisch zum
Nachschlagen oder wenn du dir ein eigenes Notebook bauen willst.

## Der Arbeitsablauf

Importiere in einem Notebook zuerst die Funktionen, die du brauchst:

```python
from stellar_cluster_finder import (
    load_fits,
    save_parquet,
    load_parquet,
    add_distance,
    convert_to_galactic,
    plot_histogram,
    plot_and_save,
    plot_3d,
    select_range,
    select_ellipse,
    find_one_cluster,
    add_absolute_magnitude,
    load_isochrones,
    get_isochrone,
)
```

**1. Daten laden.** Lies die heruntergeladene FITS-Datei ein und speichere sie als
Parquet-Datei — dieses Format lädt viel schneller, du musst die FITS-Datei also nur einmal
lesen.

```python
sterne = load_fits("data/meine_gaia_region.fits")
save_parquet(sterne, "data/meine_gaia_region.parquet")

sterne = load_parquet("data/meine_gaia_region.parquet")  # ab jetzt diese Zeile
```

Was du jetzt hast, ist das, was Gaia gemessen hat: den Ort jedes Sterns am Himmel (`ra`,
`dec`), seine Parallaxe, wie schnell er über den Himmel wandert (`pmra`, `pmdec`) und, bei
manchen Sternen, wie schnell er sich auf uns zu oder von uns weg bewegt (`radial_velocity`).

**2. Aus der Parallaxe eine Entfernung machen.** Je näher ein Stern, desto stärker scheint er
im Laufe eines Jahres hin und her zu wandern. Diese Wanderung ist die Parallaxe, und daraus
folgt die Entfernung direkt:

```python
sterne = add_distance(sterne)  # ergänzt distance_pc
```

Du bekommst eine Warnung, dass einige Zeilen auf `NaN` gesetzt wurden. Das ist normal: Die
Entfernung ist `1000 / Parallaxe`, und das ergibt keinen Sinn, wenn Gaia eine Parallaxe von
null oder weniger gemessen hat. Diese Sterne bleiben in der Tabelle, bekommen aber keine
Entfernung.

**3. Den Haufen mit dem Auge suchen — in den Messwerten selbst.** Die Mitglieder eines Haufens
sind ungefähr gleich weit entfernt, stehen dicht beieinander am Himmel und wandern gemeinsam
über ihn hinweg. Das sind drei verschiedene Blicke auf dieselben Sterne, und in jedem triffst
du eine Auswahl:

```python
plot_histogram(sterne, "distance_pc", bins=60, xlabel="Entfernung (pc)", xlim=(0, 500))
select_range(sterne, "distance_pc", minimum=130, maximum=160, selection_column="nach_entfernung")

himmel = {"x0": 56.5, "y0": 24.0, "width": 4, "height": 4, "angle": 0}
plot_and_save(sterne, "ra", "dec", title="Am Himmel", xlabel="RA (Grad)", ylabel="Dec (Grad)", ellipse_params=himmel)
select_ellipse(sterne, "ra", "dec", himmel, selection_column="nach_himmel")

bewegung = {"x0": 20, "y0": -10, "width": 8, "height": 8, "angle": 0}
plot_and_save(
    sterne,
    "pmra",
    "pmdec",
    title="Eigenbewegung",
    xlabel="pmRA (mas/Jahr)",
    ylabel="pmDec (mas/Jahr)",
    ellipse_params=bewegung,
    xlim=(0, 40),  # beide Achsen heranzoomen, um an den Klumpen zu kommen
    ylim=(-70, -10),
)
select_ellipse(sterne, "pmra", "pmdec", bewegung, selection_column="nach_bewegung")

sterne["mein_haufen"] = sterne["nach_entfernung"] & sterne["nach_himmel"] & sterne["nach_bewegung"]
```

Jede Auswahl gibt aus, wie viele Sterne sie erwischt hat. Zeichne die Darstellung noch einmal
mit `color_col="nach_bewegung"`, um zu sehen, was du tatsächlich ausgewählt hast, ändere die
Zahlen und führe sie erneut aus — diese Schleife ist der größte Teil der Arbeit.

Bei einem ganzen Gaia-Feld zeichnen die Streudiagramme die Punkte von sich aus durchsichtig und
geben die gewählte Transparenz aus. Tausende deckende Punkte verdecken einander und machen aus
dem Haufen eine ausgefüllte Fläche; durchsichtige Punkte summieren sich dort, wo die Sterne
dicht stehen, und so wird die Verdichtung sichtbar. Mit `alpha=1` schaltest du das ab und
siehst den Unterschied, mit einem kleineren Wert verstärkst du den Effekt.

`xlim` und `ylim` schneiden nur die Ansicht zu, an der Tabelle ändern sie nichts — und bei der
Eigenbewegung wirst du sie brauchen. Einige wenige schnelle Sterne ziehen beide Achsen über
Hunderte von Einheiten auseinander, sodass alles Übrige in eine Ecke gedrängt wird; zoome beide
zugleich heran, und der Haufen tritt als Knoten hervor. Die Transparenz wird dann erneut aus
dem bestimmt, was im Ausschnitt übrig bleibt — ein enger Ausschnitt wird also nicht so blass
wie das ganze Feld.

Dass du das vor jeder Umrechnung machst, hat einen Grund: Diese fünf Spalten hat Gaia
tatsächlich gemessen, fast jeder Stern hat sie. Die Positionen und Geschwindigkeiten aus dem
nächsten Schritt haben überall dort Lücken, wo eine Messung gefehlt hat.

**4. Jetzt ausrechnen, wo die Sterne wirklich stehen.** Aus Ort am Himmel, Entfernung und
Bewegung werden echte Positionen im Raum (`X`, `Y`, `Z` in Parsec) und echte Geschwindigkeiten
(`U`, `V`, `W` in km/s), sodass du deine Auswahl aus jeder Richtung ansehen kannst:

```python
sterne = convert_to_galactic(sterne)

%matplotlib widget

plot_3d(sterne, "X", "Y", "Z", title="Meine Auswahl im Raum",
        xlabel="X (pc)", ylabel="Y (pc)", zlabel="Z (pc)", color_col="mein_haufen")
```

Die Zeile `%matplotlib widget` macht die Darstellung interaktiv — ohne sie bekommst du ein
starres Bild, das du nicht drehen kannst. Hält deine Auswahl auch in der Tiefe zusammen? Am
Himmel musste sie kompakt aussehen, das war ja dein Auswahlkriterium.

Auch diese Darstellung zeichnet ihre Punkte durchsichtig, und in drei Dimensionen fällt das
noch stärker ins Gewicht als in zweien: Deckende Punkte verbergen alles, was hinter ihnen
liegt, sodass ein ganzes Feld zu einem einzigen ausgefüllten Körper wird — durchsichtige lassen
den Haufen als dunkleren Knoten darin durchscheinen. Mit `alpha=1` siehst du, wie es sonst
aussähe.

Hier gibt es mehr `NaN` als in Schritt 2: Gaia hat nur für einen Teil seiner Sterne eine
Radialgeschwindigkeit gemessen, und ohne sie gibt es kein `U`, `V`, `W`.

**5. Den Computer suchen lassen — jedes Mal mit anderen Spalten.** Der Algorithmus sieht immer
nur die Spalten, die du ihm gibst. Lass ihn deshalb mehrmals laufen, jedes Mal mit etwas
anderem, und gib jedem Durchlauf einen eigenen Namen, damit die Ergebnisse nebeneinander
stehen bleiben:

```python
# nur, wo die Sterne am Himmel stehen -- das, was ein Bild vom Himmel zeigt
find_one_cluster(sterne, columns=["ra", "dec"], cluster_label_column="himmel")

# ihre wirkliche Position im Raum, mit der Entfernung als dritter Achse
find_one_cluster(sterne, columns=["X", "Y", "Z"], cluster_label_column="position")

# alles auf einmal: der volle sechsdimensionale Phasenraum
find_one_cluster(sterne, columns=["X", "Y", "Z", "U", "V", "W"], cluster_label_column="phasenraum")
```

Die Regler — `min_cluster_size` bei HDBSCAN, `min_samples` und `eps` bei DBSCAN — fehlen hier
mit Absicht. Ohne Angabe bestimmt die Funktion sie aus den Daten, die du ihr gibst, und schreibt
die gewählten Werte in die Ausgabe. So funktioniert derselbe Aufruf für eine kleine Testtabelle
wie für einen Katalog mit Zehntausenden Sternen. Gerade `eps` kann gar keinen festen Wert haben:
Es ist eine Länge in den Einheiten der Spalten, die du ausgewählt hast, und ein halbes Grad am
Himmel ist etwas völlig anderes als ein halbes Parsec in `X, Y, Z`. Nimm die ausgegebenen Werte
als Startpunkt und ändere sie — genau darum geht es hier.

Zwei Dinge sagt der Code ausdrücklich, weil beide sonst wie ein Ergebnis aussehen. Wenn die
gefundene Gruppe einen großen Teil deiner Sterne umfasst, ist sie das Feld selbst und kein
Haufen: In diesen Spalten haben sich die Sterne nicht getrennt. Und wenn die Gruppe *genau*
`min_cluster_size` Sterne enthält, kommt diese Zahl aus deiner Einstellung und nicht aus den
Daten — führe den Durchlauf mit einem anderen Wert noch einmal aus und sieh nach, ob die Gruppe
mitwächst.

Jeder Durchlauf ergänzt eine Spalte — `himmel_HDBSCAN`, `position_HDBSCAN`,
`phasenraum_HDBSCAN`. Jeder Stern bekommt darin eine Zahl: `0`, `1`, `2`, … für die gefundenen
Gruppen und **`-1` für Sterne, die zu gar keiner Gruppe gehören**. Wenn du einem Durchlauf
keinen eigenen Namen gibst, überschreibt der nächste den vorherigen; der Code warnt dich,
bevor das passiert.

**6. Die Durchläufe vergleichen und die Auswahl schärfen.** Das ist der spannende Teil. Die
drei Durchläufe werden sich nicht einig sein, und gerade darin steckt die Aussage:

```python
plot_3d(
    sterne,
    "X",
    "Y",
    "Z",
    title="Nur am Himmel gruppiert, nach der wirklichen Position dargestellt",
    xlabel="X (pc)",
    ylabel="Y (pc)",
    zlabel="Z (pc)",
    color_col="himmel_HDBSCAN",
)
```

Stelle jeden Durchlauf so dar, wie du auch von Hand ausgewählt hast: erst das Bild ohne Farbe,
dann dasselbe Bild eingefärbt nach dem Ergebnis. Ohne das erste Bild kannst du nicht
beurteilen, ob der Algorithmus die Struktur gefunden hat, die du selbst ausgewählt hättest.

Jede Gruppe bekommt eine eigene Farbe **und** eine eigene Markerform, und das Histogramm
schraffiert seine Balken auf dieselbe Weise. Das ist Absicht: Ein Bild, das Gruppen nur über die
Farbe trennt, zerfällt in einer Schwarz-Weiß-Kopie und für alle mit einer Farbsehschwäche —
deshalb wiederholt die Form, was die Farbe sagt.

Setze für `color_col` auch `"position_HDBSCAN"` oder `"phasenraum_HDBSCAN"` ein und sieh noch
einmal hin. Wo wählen die drei Gruppierungen dieselben Sterne aus, und wo gehen sie
auseinander? Nimm das, was du siehst, mit zurück zu Schritt 3, verbessere deine eigene Auswahl
und lass Schritt 5 erneut laufen. Genau so läuft echte Auswertung ab — in mehreren Runden, von
denen jede etwas besser informiert ist als die vorige. Sterne mit dem Label `-1` werden grau
gezeichnet.

Zwei Durchläufe lohnen sich zusätzlich, wenn du Zeit hast: `["pmra", "pmdec"]`, die
Eigenbewegung direkt aus dem Katalog, und `["U", "V", "W"]`, die echten Geschwindigkeiten.
Stelle den zweiten in `X`, `Y`, `Z` dar — gruppiert nach der Bewegung, dargestellt nach der
Position. Wenn die farbigen Sterne auch im Raum zusammenliegen, sind sich zwei unabhängige
Messungen einig, und das ist ein viel stärkeres Argument als jede für sich allein. Lass die
Eigenbewegung außerdem durch alle drei Verfahren laufen — `mode="HDBSCAN"`, `mode="DBSCAN"`
und `mode="GMM"`. Die Feldsterne bilden dort eine breite Verteilung statt eines Klumpens mit
Rand, und welches der beiden dichtebasierten Verfahren damit zurechtkommt, hängt vom Haufen
ab. Das GMM ist der interessante dritte Fall: Es legt `n_cluster` Glockenkurven über die Daten
und gibt jedem Stern die, zu der er am besten passt — hier kannst du zusehen, wie diese
Annahme auf Daten trifft, die sich nicht daran halten. Sieh dir an, wo die Grenze zwischen
seinen Gruppen verläuft, und beachte, dass `n_cluster` der einzige Regler ist, den dir niemand
aus den Daten bestimmen kann.

**7. Wie alt ist er?** Die Sterne eines Haufens sind gemeinsam entstanden. Trägt man
Farbe gegen Helligkeit auf, liegen sie deshalb auf **einer** Linie — und diese Linie ändert mit
dem Alter ihre Form, weil schwere Sterne zuerst ausbrennen. Berechnete Linien eines bestimmten
Alters heißen Isochronen; lade sie dir vom [CMD-Formular](http://stev.oapd.inaf.it/cmd) herunter,
mit dem photometrischen System auf Gaia DR2 (Evans et al. 2018) und einem weiten Altersbereich,
und lege sie über deine eigenen Sterne.

Zwei Dinge müssen dafür stimmen. Aus der Helligkeit deiner Sterne muss die Entfernung
herausgerechnet werden, sonst lässt sie sich nicht mit einem Modell vergleichen. Und die y-Achse
muss auf dem Kopf stehen, denn eine Magnitude zählt rückwärts: Je kleiner die Zahl, desto heller
der Stern.

```python
haufen = add_absolute_magnitude(sterne[sterne["meine_auswahl"]])
isochronen = load_isochrones("data/isochronen.dat")

for alter in [40, 100, 250, 1000, 2500]:  # Millionen Jahre
    linie = get_isochrone(isochronen, alter)
    plot_and_save(
        haufen,
        "bp_rp",
        "abs_g_mag",
        title=f"{alter} Millionen Jahre",
        xlabel="BP - RP",
        ylabel="absolute Helligkeit G",
        invert_yaxis=True,
        line_x=linie["bp_rp"],
        line_y=linie["Gmag"],
        line_label=f"{alter} Mio. Jahre",
    )
```

Dafür brauchst du zwei Spalten aus deinem Gaia-Download, die bisher nicht benutzt worden:
`phot_g_mean_mag` und `bp_rp`. Die jüngsten Linien liegen am roten, schwachen Ende über deinen
Sternen, die ältesten knicken viel zu früh ab — dazwischen hast du das Alter eingegrenzt, ohne
es vorher gekannt zu haben. Wie eng das gelingt, hängt davon ab, wie viele helle Sterne du hast,
und darüber lohnt es sich nachzudenken, statt darüber hinwegzugehen.

## Zum Ausprobieren

Hier gibt es keine einzig richtige Antwort, und du sollst auch keine suchen. Es geht darum,
ein Gefühl dafür zu bekommen, was jede Methode sehen kann und was nicht.

- Welche Messgröße trennt die Gruppe am deutlichsten heraus — Entfernung, Ort am Himmel oder
  Eigenbewegung? Woran könnte das liegen?
- Ändere `min_cluster_size` in Schritt 5, ausgehend von dem Wert, der für dich ausgegeben
  wurde. Was passiert mit der Anzahl der Sterne in deiner Gruppe, und ab wann ergibt das
  Ergebnis keinen Sinn mehr? Gibt es einen Bereich, in dem sich kaum etwas ändert — und wie
  viel mehr würdest du einem Wert aus dessen Mitte trauen als einem vom Rand?
- Vergleiche die drei Algorithmen: `mode="HDBSCAN"`, `mode="DBSCAN"` und `mode="GMM"` legen
  jeweils eine eigene Spalte an, sodass du sie nebeneinander darstellen kannst. Wo sind sie
  sich uneinig?
- Jeder der drei Durchläufe sieht mehr als der vorige: erst den Himmel, dann den Himmel plus
  die Entfernung, dann alles einschließlich der Bewegung. Wird das Ergebnis mit jedem Schritt
  wirklich besser?
- Der Phasenraum-Durchlauf hat alle sechs Spalten und damit die meiste Information. Liefert er
  auch das beste Ergebnis? Sieh dir vorher an, über welchen Bereich `X, Y, Z` streuen und über
  welchen `U, V, W`.
- Wie viele Sterne hast du in Schritt 3 von Hand ausgewählt, und wie viele hat der Algorithmus
  gefunden? Sieh dir die an, bei denen ihr euch uneinig seid — wer hat wohl recht, und wie
  könntest du das überprüfen?

## Woran das hier getestet wurde

Alles hier wurde mit echten Daten durchgerechnet, deshalb sei auch deutlich gesagt, mit welchen
und wo der Ansatz trägt. Durchweg Gaia DR3, ein Kreis mit 4° Radius, `parallax BETWEEN 3 AND 12`:

| Haufen | Mittelpunkt (ra, dec) | Sterne |
| --- | --- | --- |
| Plejaden (Melotte 22) | 56.60, +24.11 | 15.540 |
| Praesepe / Bienenkorb (NGC 2632) | 130.05, +19.62 | 12.216 |

Die Isochronen sind PARSEC-Modelle vom [CMD-3.9-Formular](http://stev.oapd.inaf.it/cmd), mit
Gaia-DR2-Photometrie nach Evans et al. 2018, solarer Metallizität und `log(Alter/Jahre)` von
7,0 bis 9,6 in Schritten von 0,2 — 14 Alter zwischen 10 Millionen und 4 Milliarden Jahren.

**Was bei beiden Haufen funktioniert.** Das Auswählen von Hand durchgehend: Keiner der
Startwerte in Notebook 1 ist auf einen bestimmten Haufen zugeschnitten, sie werden alle aus den
Daten abgelesen. Clustern auf `X`, `Y`, `Z` mit HDBSCAN. Clustern auf der Eigenbewegung mit
DBSCAN. Clustern auf `U`, `V`, `W` — mit der Einschränkung, dass nur etwa ein Fünftel der Sterne
überhaupt eine Radialgeschwindigkeit hat, dieser Durchlauf also mit viel weniger Sternen
arbeitet als die anderen.

**Was nicht funktioniert, und warum man das wissen sollte.**

- **Clustern auf `ra`, `dec` liefert bei beiden Haufen das ganze Feld.** Das ist Absicht und das
  Notebook baut darauf auf: Am Himmel ist ein Haufen eine leichte Verdichtung und keine Insel,
  und ein dichtebasiertes Verfahren findet dort keine Lücke. Der Code warnt dich, wenn es
  passiert.
- **HDBSCAN auf der Eigenbewegung funktioniert bei den Plejaden und scheitert bei Praesepe.**
  Die Plejaden liegen rund 1,0σ neben der Verteilung der Feldsterne, Praesepe nur 0,8σ, und
  unterhalb davon gibt HDBSCAN das Feld zurück. DBSCAN findet beide. Wenn die Bewegung deines
  Haufens wenig auffällig ist, geht es dir genauso.
- **Der sechsdimensionale Durchlauf liefert bei beiden Haufen genau `min_cluster_size` Sterne**
  — die Spitze einer einzelnen Verdichtung und keinen Haufen. Er warnt davor, und die Zahl kommt
  aus der Einstellung und nicht aus einer Messung.
- **Die Plejaden lassen sich nicht genau datieren, und das liegt an den Daten.** Oberhalb von
  etwa `G = 3` übersteuert Gaia, die hellsten Mitglieder fehlen also, und damit ist der
  Abknickpunkt — der einzige wirklich altersempfindliche Teil des Diagramms — fast leer. Die
  Plejaden lassen sich grob auf 60 bis 250 Millionen Jahre eingrenzen, der Literaturwert liegt
  bei 125. Praesepe ist älter und knickt dort ab, wo Gaia gut misst: 631 bis 1000 Millionen
  Jahre, Literaturwert etwa 700 bis 800. Ein älterer Haufen lässt sich besser datieren.

**Nicht getestet:** jeder andere Haufen, jede andere Durchmusterung als Gaia DR3, Isochronen in
einem anderen photometrischen System oder mit anderer Metallizität, sowie Downloads ohne
Parallaxen-Grenzen. Der Code sollte damit zurechtkommen, überprüft hat es aber niemand.

## Die Funktionen

| Funktion | Was sie macht |
| --- | --- |
| `load_fits(filename)` | Eine FITS-Tabelle (ein Gaia-Download) als Tabelle einlesen |
| `load_parquet(filename)` | Eine Parquet-Datei als Tabelle einlesen |
| `save_parquet(dataframe, filename)` | Eine Tabelle als Parquet-Datei speichern |
| `add_distance(df)` | Aus der Parallaxe die Entfernung `distance_pc` ergänzen |
| `convert_to_galactic(df)` | Positionen `X, Y, Z` und Geschwindigkeiten `U, V, W` ergänzen |
| `plot_histogram(df, column)` | Zeigen, wie eine Größe verteilt ist, mit `xlim` zum Heranzoomen |
| `plot_dataframe(df, x_col, y_col, …)` | Ein Streudiagramm erzeugen und zurückgeben |
| `plot_and_save(df, x_col, y_col, …)` | Streudiagramm, wahlweise mit Ellipse oder Linie |
| `plot_ellipse(params)` | Eine Ellipse in die gerade erzeugte Darstellung zeichnen |
| `plot_3d(df, x_col, y_col, z_col, …)` | Streudiagramm in drei Dimensionen |
| `select_range(df, column, minimum, maximum)` | Sterne zwischen zwei Grenzen markieren und ihre Anzahl ausgeben |
| `select_ellipse(df, x_col, y_col, params)` | Sterne innerhalb einer gezeichneten Ellipse markieren und ihre Anzahl ausgeben |
| `find_one_cluster(df, columns)` | Haufen finden mit HDBSCAN, DBSCAN oder Gaussian Mixture Model |
| `add_absolute_magnitude(df)` | `abs_g_mag` ergänzen: die Helligkeit ohne den Einfluss der Entfernung |
| `load_isochrones(filename)` | Eine PARSEC/CMD-Isochronendatei einlesen |
| `get_isochrone(isochrones, age_myr)` | Die Isochrone eines Alters herausgreifen, fertig zum Zeichnen |

Jede Funktion erklärt ihre Argumente in ihrem eigenen Docstring — allerdings auf Englisch,
so wie es in der Programmierung üblich ist. Schreibe im Notebook `help(find_one_cluster)`
oder setze ein `?` hinter den Namen, um ihn zu lesen.
