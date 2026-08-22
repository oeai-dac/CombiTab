# CombiTab 2

Kombinationstabellen, Seriation und Korrespondenzanalyse für die archäologische
Auswertung — als Desktop-Anwendung für Windows, macOS und Linux und als
Web-Anwendung, die vollständig auf dem eigenen Rechner läuft.

Vollständiger Arbeitszyklus: **laden → seriieren/analysieren → annotieren →
Robustheit prüfen → exportieren**. React + TypeScript + Vite im ÖAI-Design, mit
einem selbst geschriebenen WebGL2-Renderer für die Matrix.

MIT-Lizenz · © Christian Gugl / Österreichisches Archäologisches Institut (ÖAW)

---

## Herunterladen

**Ohne Installation ausprobieren:** <https://oeai-dac.github.io/CombiTab/>

**Installieren:** Die fertigen Pakete stehen unter
[**Releases**](https://github.com/oeai-dac/CombiTab/releases).

| Ihr System | Datei |
|---|---|
| Windows 10/11 | `CombiTab-*-Setup-x64.exe` — oder `*-portable-x64.exe` ohne Installation |
| macOS (Apple M1–M4) | `CombiTab-*-arm64.dmg` |
| macOS (Intel) | `CombiTab-*-x64.dmg` |
| Linux, universell | `CombiTab-*-x86_64.AppImage` |
| Debian, Ubuntu, Mint | `combitab_*_amd64.deb` |
| Fedora, openSUSE, RHEL | `combitab-*.x86_64.rpm` |

Die Pakete sind nicht digital signiert; Windows und macOS zeigen deshalb beim
ersten Start eine Warnung. **[docs/INSTALLATION.md](docs/INSTALLATION.md)**
führt Schritt für Schritt hindurch — auch ohne technische Vorkenntnisse.

---

## Grundsätze

- **Alles bleibt lokal.** Es gibt kein Backend. Daten werden nie hochgeladen; die
  App ist installierbar und nach dem ersten Besuch offline lauffähig. Die
  Desktop-Fassung geht weiter: Eine Content-Security-Policy unterbindet dort
  jede Verbindung nach außen, und selbst die Schriften sind mitgeliefert statt
  von einem CDN geladen.
- **Keine Fremdbibliotheken für die Fachlogik.** SVD, Korrespondenzanalyse,
  PDF-Erzeugung, Internationalisierung, Farbmetrik und Icon-Erzeugung sind
  selbst geschrieben. Einzige Laufzeit-Abhängigkeiten sind React und — nur bei
  tatsächlicher XLSX-Nutzung nachgeladen — SheetJS.
- **Grenzen werden benannt, nicht kaschiert.** Wo ein Budget verfehlt wird, wo
  eine Zahl ein Ziel und kein gemessenes Ergebnis ist, und wo das Werkzeug etwas
  bewusst *nicht* behauptet, steht das im Text — siehe „Ehrliche Grenzen".

## Aus dem Quelltext bauen

```bash
npm ci
npm run dev        # Entwicklungsserver (Vite)
npm run build      # Typprüfung + Produktions-Build nach dist/
npm run preview    # Build lokal ansehen
npm test           # gesamte Testsuite
npm run electron   # Build + Start im Desktop-Gehäuse
npm run smoke      # Selbsttest der Desktop-Fassung
npm run dist       # Installationspakete für das aktuelle System
npm run bench      # Performance-Benchmark
npm run validate   # nur die CA-Referenzvalidierung
npm run gen-icons  # Icons neu erzeugen (PWA + Paket-Icon)
```

Einzelheiten zum Paketbau, zur Freigabe und zur Code-Signierung:
**[docs/BUILD.md](docs/BUILD.md)**.

Beim Start wird ein Demo-Datensatz geladen. Eigene Dateien per **Drag & Drop**
oder „Datei laden…": v1-Projekt (`.json`, wird migriert), v2-Projekt (`.json`),
Rohtabelle (`.csv`/`.tsv`/`.xlsx`).

---

## Neu in Version 2.0

- **Materialgruppen sind frei verwaltbar.** Im Reiter *Metadaten &
  Materialzuweisung* lassen sich Gruppen **anlegen, umbenennen, umfärben und
  löschen**. Neue Gruppen erscheinen unmittelbar in der Zuweisungsliste, in der
  Matrix-Legende, in allen Bild-Exporten und als Filter-Chip in der Matrix-Ansicht.
  Beim Löschen wandern die zugewiesenen Typen in die erste verbleibende Gruppe und
  übernehmen deren Farbe — es entsteht keine verwaiste Zuweisung; die letzte
  verbleibende Gruppe ist nicht löschbar, damit jeder Typ stets eine gültige Farbe
  hat. Die Farbe einer neuen Gruppe wird nicht zyklisch vergeben, sondern als jene
  Palettenfarbe gewählt, die von allen bereits vergebenen perzeptuell am weitesten
  entfernt ist.
- **Bild-Export korrigiert.** Zwei Layoutfehler der Vorgängerfassung sind behoben:
  - Die gedrehten Spaltenbeschriftungen liefen wegen einer gegenüber dem
    Textanker invertierten Drehrichtung **in die Matrix hinein**. Sie stehen jetzt
    in dem für sie reservierten Rand über der Matrix und lesen von unten nach oben.
  - Die **Legende ging nicht in die Breite der Zeichenfläche ein**, sodass die
    letzten Materialgruppen rechts abgeschnitten wurden. Sie bricht jetzt um, und
    die Zeichenfläche wächst mit.

  Zusätzlich rechnet das Layout jetzt mit **echten Helvetica-Textbreiten** statt
  mit einer pauschalen Zeichenbreite — also mit der Metrik der Schrift, die der
  PDF-Export tatsächlich setzt. Überlange Bezeichnungen werden sichtbar gekürzt,
  statt den Rand zu sprengen. SVG, PNG und PDF stammen weiterhin aus **einer**
  gemeinsamen Szene und sind dadurch deckungsgleich.
- **Neue Testsuiten:** `core/materialGroups.test.ts` (Gruppenverwaltung, inklusive
  der Invariante „jeder Typ zeigt auf eine existierende Gruppe und trägt deren
  Farbe") und `export/layout.test.ts` (Export-Geometrie: kein Text ragt über die
  Zeichenfläche hinaus, kein Text überlappt eine gefüllte Fläche).

---

## Funktionsumfang

### Matrix

- **WebGL2-Rendering** der Kombinationstabelle, getrieben vom `ProjectV2`:
  Zellfarbe = Materialgruppenfarbe der Spalte, Sättigung nach Häufigkeit.
- **Zoom** (Mausrad, um den Cursor) und **Verschieben** (Ziehen der Fläche),
  bedienbar per Maus wie per Touch (Pointer-Events).
- **Hover-Inspektor** und **Klick-Selektion** von Zelle / Kontext / Typ mit
  Kreuz-Hervorhebung.
- **Drag-Reorder:** Zeilen-/Spaltenbeschriftung ziehen ordnet Kontexte bzw. Typen
  manuell um. Umsortieren ist **O(1)** — es wird nur eine kleine
  Order-Lookup-Textur neu hochgeladen, nicht die Zellen.
- **Fixierte Elemente:** Kontexte/Typen im Inspektor fixieren. Fixierte Elemente
  behalten ihre Position; die Seriation ordnet nur die freien Elemente in die
  Lücken. Gespiegelt in `rowMetadata.isFixed` / `columnMetadata.isFixed`.
- **Undo/Redo** über Momentaufnahmen von Ordnung + Fixierungen (Strg/⌘+Z,
  Strg/⌘+Umschalt+Z bzw. Strg+Y).
- **Übersichts-Minimap** mit Viewport-Rechteck für schnelle Navigation in großen
  Matrizen.
- **Canvas-2D-Ersatzpfad:** Fehlt WebGL2, zeichnet derselbe Renderer hinter
  demselben Interface über einen 2D-Kontext — mit exakt nachgebildetem Farbmodell
  (`matrix/cellColor.ts`). `renderer.backend` liefert `"webgl2" | "canvas2d" | "none"`.

### Seriation & Analyse

- **Drei Verfahren:** Schwerpunktmethode (reziprokes Mittel), CA-basierte
  Seriation und iterative Optimierung — mit Live-Qualitäts-Score (Konzentration,
  Kontinuität, Anti-Robinson, Gesamt).
- **Korrespondenzanalyse:** exakte, SVD-basierte CA (`analysis/ca.ts`, einseitiges
  Jacobi in `analysis/svd.ts`) mit **Biplot** (Dim 1 × 2) und **Scree-Plot**.
- **Ford-Diagramm:** „Battleship curves" — Balkenbreite = Anteil des Typs am
  Inventar des Kontexts, Kontexte in aktueller Seriationsreihenfolge.
- **Bootstrap-Stabilität:** multinomiales Resampling der Typhäufigkeiten je
  Kontext; pro Wiederholung CA-Dimension 1, am Vorzeichen der Referenz
  ausgerichtet. **Caterpillar-Plot** mit 90 %-Rangintervall, Median- und
  Referenzrang; globale Stabilität als mittleres |Spearman-ρ|. 100/200/500
  Wiederholungen, reproduzierbar über einen mulberry32-Seed.
- **Brushing & Linking:** eine gemeinsame Auswahl-/Hover-Ebene (`link.tsx`) koppelt
  Matrix, Biplot und Ford; die Auswahl bleibt beim Reiterwechsel erhalten.

### Filter & Fokus

Materialgruppen-Chips, Bereichsfilter über Zeilen/Spalten, „leere Zeilen/Spalten
aus" und ein **Fokus-Modus** auf die Nachbarschaft der Auswahl. Die gefilterte
Sicht wird von allen Ansichten identisch konsumiert; Umsortieren, Seriation,
Fixieren und Annotieren wirken auf die sichtbare Teilmenge und werden ins
Grundprojekt zurückgeschrieben. Aktive Filter gehen in Teilen-Link, Autosave und
Projekt-Export ein.

### Annotationen & Metadaten

- **Zell-Annotationen** (Modus „Annotieren"): Zellbereich aufziehen und als Batch
  setzen — Sicherheit (sicher/unsicher/fraglich), Fragmentierung,
  Stückzahl-Bereich, Inventarnummern, Notizen. Annotierte Zellen tragen einen
  Ampel-Marker. Beim Anwenden auf eine gemischte Auswahl werden **nur tatsächlich
  angefasste Felder** gesetzt; uneinheitliche Felder bleiben unverändert erhalten.
  Ein Feld explizit zu leeren löscht es weiterhin.
- **„Nicht erfasst" vs. strukturelle Absenz:** Eine Zelle kann als fehlender Wert
  markiert werden, klar getrennt von der echten 0. Solche Zellen zeichnet die
  Matrix mit einem diagonalen Kreuz; der Inspektor zeigt eine Präsenzstatistik
  (vorhanden / nicht vorhanden / nicht erfasst). Der Import erkennt `?`, `NA`,
  `N/A`; der Export schreibt `?` zurück.
- **Metadaten-Reiter:** Materialgruppen-Verwaltung (siehe oben) und eine
  durchsuchbare Typentabelle zum Zuweisen der Gruppe und Markieren von Leittypen.

### Farbmetrik & Barrierefreiheit

- **CVD-sichere Palette** auf Knopfdruck (Okabe-Ito), Vorschau unter
  Deuteranopie/Protanopie/Tritanopie und eine Warnung vor kaum unterscheidbaren
  Farbpaaren (perzeptueller Abstand ΔE unter der Schwelle). Farbkern
  `core/palette.ts`: WCAG-Kontrast, Dichromasie-Simulation (Viénot 1999, korrekt
  im linearen Licht), CIELAB-ΔE76.
- **Tastatur & Screenreader:** Skip-Link, `:focus-visible`-Fokusring, Reiter nach
  dem WAI-ARIA-Tabs-Muster (Roving-`tabindex`, Pfeiltasten/Home/End), Landmarken,
  `role="img"` mit beschreibendem Label auf der Matrix-Canvas, `role="alert"` für
  Fehler, `<html lang>` folgt der UI-Sprache.
- **`prefers-reduced-motion`** schaltet nicht-essenzielle Übergänge ab.

### Export

Über das Export-Menü in der Kopfleiste, aus jedem Reiter erreichbar:

- **Bild der Matrix** in aktueller Anzeige-Reihenfolge: **SVG** und **PDF** (echter
  Vektor ohne Fremdbibliothek — PDF direkt als Content-Stream mit
  Base-14-Helvetica) sowie **PNG** (×2 rasterisiert). Fixierte Elemente sind rot
  beschriftet, Materialgruppen erscheinen als Legende.
- **Sortierte Daten:** CSV und XLSX in Seriationsreihenfolge.
- **Projektdatei:** v2 (kanonisches Schema, `.combitab.json`) und über den
  Migrationsadapter v1 (abwärtskompatibel).
- **Methods-Bericht** als Markdown, **zweisprachig**, mit Datensatzkennzahlen,
  Verfahren, Qualitätsmetriken, CA-Trägheit, Bootstrap-Stabilität und Zitation.
- **Linked Open Data** als Turtle und JSON-LD (siehe unten).

Cross-Browser gehärtet: CSV mit UTF-8-BOM für Excel unter Windows, SVG mit
XML-Prolog und Namespace, PDF mit `/MediaBox`, transliterierte portable
Dateinamen und ein `window.open`-Rückfall für Browser ohne `<a download>`.

### Linked Open Data (CIDOC-CRM / CRMarchaeo)

- Kontexte werden als `crmarchaeo:A2_Stratigraphic_Volume_Unit` modelliert (nicht
  als Sites); der **eine** `crm:E27_Site` ist die Fundstelle. Je Kontext eine
  `crmarchaeo:A1_Excavation_Processing_Unit` mit
  `crmarchaeo:AP5_removed_part_or_all_of`.
- **Echte Objektidentitäten:** Eine Zelle mit Häufigkeit n erzeugt n einzelne
  `crm:E22_Human-Made_Object`, im Kontext über `crmarchaeo:AP21_contains` verortet.
  Ein dokumentiertes Limit (`maxObjectsPerCell`, Default 1000) fällt darüber
  explizit auf ein Aggregat zurück; `objectIdentities: false` erzwingt das Aggregat.
- **Gemeinsame Zwischenrepräsentation:** Turtle und JSON-LD werden aus *einem*
  IR-Graphen gerendert und tragen garantiert dieselben Aussagen.
- „Nicht erfasst" **und** strukturelle Absenz erzeugen bewusst **keinen** Fund.

### Teilen, Autosave, Offline

- **Teilen-Link:** Der komplette Zustand (Projekt, Reihenfolge, Filter,
  Annotationen, Fehlwertmarkierungen, aktiver Reiter) wird per nativem
  `CompressionStream` und base64url ins **URL-Fragment** kodiert — das Fragment
  geht nicht an einen Server. Über ~16 000 Zeichen erscheint der Hinweis, statt
  dessen die Projektdatei zu teilen.
- **Autosave in IndexedDB** beim Ausblenden/Schließen der Seite; beim nächsten
  Start bietet ein Banner die Wiederherstellung an. Alle Zugriffe sind
  fehlertolerant (Privatmodus → still deaktiviert).
- **PWA:** Manifest, Service Worker (App-Shell beim Install cachen, Navigationen
  network-first mit Offline-Rückfall, übrige Anfragen stale-while-revalidate),
  Installations-Knopf. Icons werden dependency-frei per Node-`zlib` erzeugt.

### Zweisprachigkeit

Die gesamte sichtbare Oberfläche liegt in Deutsch und Englisch vor, inklusive der
Canvas-Beschriftungen. Umschaltung über den DE/EN-Knopf; die Wahl wird gespeichert
und beim Start aus der Browsersprache vorbelegt. Eigener i18n-Kern (`src/i18n/`)
mit flachem Wörterbuch, `{var}`-Interpolation und Rückfall EN → DE → Schlüssel.

---

## Desktop-Fassung

Dieselbe Anwendung, in ein Electron-Gehäuse gepackt — damit Nutzerinnen und
Nutzer ohne Node.js, Terminal oder Paketverwaltung auskommen.

- **Geladen wird über ein eigenes `app://`-Schema, nicht über `file://`.** Das
  ist kein Detail: Die Rechen-Worker sind **Modul-Worker**
  (`new Worker(url, { type: "module" })`), und Chromium blockiert deren Laden
  unter `file://` — Korrespondenzanalyse, Bootstrap und Score-Berechnung wären
  stillschweigend ausgefallen. Das eigene Schema liefert zudem einen stabilen
  Origin und damit verlässliches IndexedDB (Autosave), localStorage (Theme,
  Sprache) und einen sicheren Kontext für `CompressionStream` (Teilen-Link).
- **Abgeriegelt.** `contextIsolation`, `sandbox`, kein `nodeIntegration`. Das
  Preload-Skript reicht **keine** Datei- oder Node-Funktion durch, sondern nur
  eine Kennung, an der die Oberfläche das Gehäuse erkennt. Eine
  Content-Security-Policy verbietet jede Verbindung nach außen.
- **Exporte landen dort, wo Sie es wollen:** Der `<a download>`-Pfad der App
  öffnet im Gehäuse einen nativen Speichern-Dialog statt wortlos im
  Download-Ordner abzulegen.
- **Selbsttest statt Vermutung.** `npm run smoke` startet ein unsichtbares
  Fenster und prüft die zehn Eigenschaften, die beim Wechsel ins Gehäuse brechen
  können — Modul-Worker, WebGL2, sicherer Kontext, geladene Schriften, keine
  externen Ressourcen. Er läuft in der CI mit und wurde auch gegen das fertig
  gepackte AppImage ausgeführt.

---

## Architektur

```
src/
  core/                 Modell, Migration, Import, Filter, Fehlwerte,
                        Materialgruppen, Palette, Theme, Autosave, Teilen-Link
  analysis/
    svd.ts · ca.ts      dünne SVD + Korrespondenzanalyse
    bootstrap.ts        Bootstrap-Stabilität
    methods.ts          zweisprachiger Methods-Bericht
    stabilityStore.ts   Cache des letzten Bootstrap-Laufs für den Export
  export/               Szene → SVG/PNG/PDF · Textmetrik · CSV/XLSX ·
                        Projekt v2/v1 · Turtle/JSON-LD
  annotations/          Annotations-Modell (Batch, Ampelfarbe)
  matrix/               WebGL2-Renderer (Linking, Bereichsselektion, Marker)
                        + fix-bewusste Ordnungslogik + Zell-Farbmodell
  seriation/            Seriationsverfahren + Qualitätsmetriken
  workers/              Rechen-Worker (CA/Bootstrap/Seriation) und
                        Score-Worker (Qualitäts-Neuberechnung)
  i18n/                 i18n-Kern + React-Anbindung
  validation/           CA-Referenzvalidierung gegen publizierte Werte
  bench/                seed-basierter Synthese-Generator + Benchmark
  components/           Shell · MatrixView · Inspector · CAView · FordView
                        · StabilityView · ExportMenu · AnnotationEditor · MetaView
  App.tsx · theme.css   Zustand/Routing · ÖAI-Design-Tokens
  fonts/ · fonts.css    mitgelieferte Schriften (erzeugt, siehe docs/BUILD.md)

electron/
  main.js               Hauptprozess: app://-Protokoll, Fenster, Menü, Downloads
  preload.cjs           reicht nur eine Desktop-Kennung durch, keinen Systemzugriff
  smoke.js              Selbsttest der Desktop-Fassung (npm run smoke)
```

### Rechenkerne und Fehlwerte

Als „nicht erfasst" markierte Zellen werden **nicht wie 0** behandelt:

- **Metrik (Maskierung):** Die Konzentration überspringt markierte Zellen; die
  Kontinuität rechnet sie innerhalb der Belegungsspanne aus dem Nenner heraus
  (unbekannt ist keine Lücke); der Anti-Robinson-Index nutzt eine
  **paarweise-vollständige** Cosinus-Ähnlichkeit.
- **CA (Imputation):** Da CA eine vollständige Kontingenztabelle braucht, werden
  markierte Zellen iterativ mit ihrem Erwartungswert unter Unabhängigkeit gefüllt
  (EM-/Nora-Chouteau-Schema). Bewusst konservativ: unbekannte Zellen bekommen keine
  unterstellte Assoziation.
- Ohne Fehlwerte nehmen alle Kerne ihren unveränderten Schnellpfad — verifiziert,
  unter anderem bleibt die CA-Referenzvalidierung exakt erhalten.

### Score-Neuberechnung im Worker

Der Anti-Robinson-Index ist O(NR²·NC) und lief früher synchron im Haupt-Thread nach
jedem Drop/Undo/Seriation — bei 1.000×1.000 mehrere Sekunden Blockade. Er läuft
jetzt in einem **eigenen** Score-Worker, getrennt vom Rechen-Worker, damit ein
Score-Lauf keinen laufenden Bootstrap abbricht. Die Matrix wird nur beim
Matrix-/Sichtwechsel übertragen (Epoch-Cache); Folgeanfragen schicken nur die
Ordnungs-Arrays. Ohne Worker (Test-/Embed-Kontexte) rechnet der Client synchron
weiter.

---

## Tests

`npm test` führt alle Suiten aus — framework-frei, ohne Browser, ohne externe
Dateien:

| Bereich | Suiten |
|---|---|
| Modell & I/O | Ordnungsmodell, Zell-Farbmodell, Filter, Fehlwerte, Teilen-Link, Theme, Palette, **Materialgruppen** |
| Analyse | CA/SVD, Fehlwert-Kerne, Bootstrap, Methods-Bericht, CA-Referenzvalidierung |
| Export | Export, **Export-Layout**, Cross-Browser, RDF |
| Sonstiges | Seriationsmetriken, Seriationsstrategien, Score-Worker, i18n-Parität, PWA, Synthese-Generator, Annotationen |

**Aktueller Stand: 319 Prüfungen, alle bestanden.** Zusätzlich sind
Worker-`postMessage`-Grenze, Autosave-Round-Trip und der Canvas-2D-Zeichenpfad je
per Wegwerf-Attrappe headless bestätigt worden.

### CA-Referenzvalidierung

Die selbst geschriebene Korrespondenzanalyse wird gegen **publizierte** Ergebnisse
geprüft, nicht nur gegen sich selbst. Primäranker ist Greenacres „smoking"-Datensatz
(auch `ca::smoke` im R-Paket `ca`): Haupt­trägheiten **0.074759 / 0.010017 /
0.000414** auf sechs Dezimalen identisch, Trägheitsanteile 87,76 % / 11,76 %,
Prinzipalkoordinaten bis auf ≤ 0.001 (Achsen-Vorzeichen werden vorher ausgerichtet
— der bekannte Eigenvektor-Freiheitsgrad). An einer perfekten Petrie-/Robinson-
Inzidenzmatrix gewinnt CA-Dimension 1 die wahre Reihenfolge zurück (Spearman
ρ ≈ ±0,999, Referenzergebnis nach Hill 1974).

---

## Ehrliche Grenzen

- **Das GPU-Budget war als „10⁶ Zellen bei 60 fps" formuliert — diese Kennzahl ist
  nicht messbar.** Der Renderer cullt, zeichnet also nur das Sichtfenster. Wie viele
  Zellen tatsächlich gezeichnet werden, hängt an der Bildschirmgröße, nicht am
  Datensatz: Für eine Million Zellen bei der kleinsten Zellgröße (2 px) bräuchte es
  rund 2000 × 2000 CSS-Pixel Zeichenfläche. Auf üblichen Displays ist die volle
  Million schlicht nicht darstellbar. Maßgeblich ist stattdessen die **Zeichenzeit
  bei maximaler Verkleinerung**.
- **Gemessen auf der Zielhardware** — Chromebook Plus, ChromeOS, Perf-HUD mit
  WebGL2-Backend, 1.000×1.000-Datensatz, Zwei-Sekunden-Benchmark:

  | gezeichnete Zellen | Draw ø | Draw p95 |
  |---|---|---|
  | 33 712 | 0,5 ms | 0,7 ms |
  | 73 146 | 0,6 ms | 0,9 ms |
  | 264 759 | 1,1 ms | 1,7 ms |

  Die Zeichenzeit wächst linear mit der Zellzahl (≈ 4,3 µs je 1000 Zellen;
  p95 ≈ 0,57 ms + 4,28 µs pro 1000). Die Bildrate lag durchgehend bei 60 fps —
  **vsync-gedeckelt**, Frame ø = p95 = 16,7 ms = 1/60 s; die Aussagekraft liegt
  daher bei der Draw-Zeit, nicht bei der FPS-Zahl.
- **Hochgerechnet, nicht gemessen:** Auf 10⁶ gezeichnete Zellen extrapoliert ergibt
  die Gerade ≈ 3,0 ms ø und ≈ 4,8 ms p95, also rund dreifache Reserve zum
  16-ms-Frame-Budget. Das ist eine Extrapolation aus drei Punkten über einen
  8-fachen Lastbereich, deren höchster bei 26 % der Zielmenge liegt — kein
  Messergebnis. Die Zielhardware-Verifikation ist damit für die tatsächlich
  darstellbare Last erbracht; die volle Million gezeichneter Zellen bleibt aus dem
  oben genannten Grund (Bildschirmgröße) unerreichbar und damit ungemessen. Das
  Perf-HUD (⏱-Knopf oder Umschalt+P) enthält den Benchmark für eigene Läufe.
- **CPU-Budgets, gemessen auf der Zielhardware** (Chromebook Plus, ChromeOS-Linux,
  Node 24, `npm run bench`):

  | Messung | Budget | Gemessen | |
  |---|---|---|---|
  | Live-Drag-Reorder je Op (Ordnung 1000) | < 16 ms | 0,01 ms | ✓ 2018× Reserve |
  | CA 500×500 (Dim 1–4) | < 2000 ms | 3349 ms | ⚠ 1,7× über Budget |
  | Score-Neuberechnung 200×200 | < 16 ms | 16,5 ms | ⚠ punktgenau am Limit |
  | Datenaufbau 10⁶ Zellen | — | 2,79 ms | |
  | Score-Neuberechnung 1.000×1.000 | — | 2083 ms | |
  | Seriation „centroid" 200×200 | — | 3,23 ms | |
  | Seriation „ca" 200×200 | — | 342 ms | |
  | Seriation „iterative" 200×200 | — | 16,3 ms | |

  Nur die algorithmische Invariante (Live-Drag) gatet den Exit-Code; sie hält mit
  großem Abstand. Die beiden Wall-Clock-Überschreitungen sind reale, benannte
  Grenzen:
  - **CA 500×500 verfehlt das 2-Sekunden-Budget um Faktor 1,7.** Sie läuft im
    Rechen-Worker, die Oberfläche friert also nicht ein — es dauert schlicht rund
    dreieinhalb Sekunden, bis Biplot und Scree stehen. Ein WASM-Pfad bleibt der
    mögliche Ausbaugrad.
  - **Score-Neuberechnung 200×200 liegt mit 16,5 ms exakt auf der Budgetlinie.**
    Da sie seit der Auslagerung im Score-Worker läuft, ist das kein Frame-Budget
    mehr, sondern die Latenz bis zur Aktualisierung der Score-Anzeige — spürbar
    erst bei großen Matrizen (1.000×1.000: rund zwei Sekunden). Der Live-Drag
    selbst bleibt davon unberührt, weil der Score erst beim Loslassen neu gerechnet
    wird.
- **Keine erfundene Stratigraphie.** Das Datenmodell erfasst **keine** beobachteten
  Harris-Matrix-Relationen. Der LOD-Export behauptet deshalb keine
  `crmarchaeo:AP13`-Relationen, sondern exportiert nur die seriationsabgeleitete
  relative Ordnung (`ctb:seriationPosition`) — ausdrücklich als inferiert markiert
  und **ohne behauptete Zeitrichtung**. Seriation liefert eine Sequenz, keine
  absolute Richtung. AP13 ist der dokumentierte Erweiterungspunkt, sobald
  stratigrafische Beobachtungen vorliegen.
- **Umsortieren ist zeigergebunden.** Eine vollständige Tastatur-Umsortierung wäre
  ein eigenes Feature; die Zellwerte sind alternativ über Inspektor und
  CSV/Excel-Export zugänglich.
- **Kein Browser-Harness.** Visual-Regression-Tests und ein vollständiger
  Screenreader-/Kontrast-Audit (z. B. Playwright + axe-core) sind headless nicht
  durchführbar und bleiben empfohlener Folgeschritt. Die Export-Geometrie wird
  stattdessen **rechnerisch** geprüft (Überlappungs- und Randtest auf der Szene),
  was die beiden in 2.0 behobenen Layoutfehler zuverlässig abfängt.
- **Annotationen sind nicht im Undo/Redo.** Der Stack umfasst Anzeige-Ordnung und
  Fixierungen; Setzen/Löschen von Zell-Annotationen läuft daneben.
- **`xlsx` (SheetJS)** hat in der npm-Fassung 0.18.5 bekannte Advisories ohne
  Fix im npm-Registry. Durch Lazy-Loading wird der Code nur bei tatsächlicher
  XLSX-Nutzung ausgeführt.
- **Die Pakete sind nicht signiert.** Windows zeigt eine SmartScreen-Meldung,
  macOS eine Gatekeeper-Warnung; beides ist in
  [docs/INSTALLATION.md](docs/INSTALLATION.md) erklärt. Für macOS wird eine
  **Ad-hoc-Signatur** gesetzt — nicht um Gatekeeper zu beruhigen, sondern weil
  eine gänzlich unsignierte Anwendung auf Apple Silicon gar nicht erst startet.
- **Kein Auto-Update.** Ohne Apple-Zertifikat arbeitet `electron-updater` auf
  macOS nicht verlässlich, und ein Versions-Check widerspräche dem Grundsatz,
  dass die Anwendung von sich aus keine Verbindung aufbaut. Neue Fassungen
  werden über die Releases-Seite bezogen.
- **Nicht auf allen Zielsystemen selbst gestartet.** Verifiziert sind der Bau
  aller Pakete in der CI sowie der Selbsttest unter Linux — einmal aus dem
  Quelltext und einmal gegen das gepackte AppImage. Die Windows- und
  macOS-Pakete sind gebaut, aber noch nicht auf der jeweiligen Zielhardware
  gestartet worden; das bleibt vor der Veröffentlichung zu tun.

---

## Mögliche nächste Schritte

- Annotationen in Undo/Redo aufnehmen; Einzelzell-Annotation im Navigieren-Modus.
- Fixierte Elemente auch bei der CA berücksichtigen; CA-Achsenwahl interaktiv.
- Gleichzeitige Nebeneinander-Ansicht (Matrix + Biplot) für Live-Brushing.
- WASM-Pfad für CA und Bootstrap sehr großer Matrizen.
- Browser-Harness für Visual-Regression und A11y-Audit.
- Zielhardware-Verifikation der PWA-Installation (GPU-Seite ist erledigt, siehe
  „Ehrliche Grenzen").
