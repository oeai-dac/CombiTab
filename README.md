# CombiTab 2

**English** · [Deutsch ↓](#combitab-2--deutsch)

Combination tables, seriation and correspondence analysis for archaeological
analysis — as a desktop application for Windows, macOS and Linux, and as a web
application that runs entirely on your own machine.

The full working cycle: **load → seriate/analyse → annotate → test robustness →
export**. React + TypeScript + Vite in the ÖAI design, with a hand-written WebGL2
renderer for the matrix.

MIT licence · © Christian Gugl / Austrian Archaeological Institute (ÖAW)

---

## Download

**Try it without installing:** <https://oeai-dac.github.io/CombiTab/>

**Install:** the finished packages are on the
[**Releases**](https://github.com/oeai-dac/CombiTab/releases) page.

| Your system | File |
|---|---|
| Windows 10/11 | `CombiTab-*-Setup-x64.exe` — or `*-portable-x64.exe`, no installation |
| macOS (Apple M1–M4) | `CombiTab-*-arm64.dmg` |
| Linux, any distribution | `CombiTab-*-x86_64.AppImage` |
| Ubuntu 22.04+, Debian 12+, Mint | `combitab_*_amd64.deb` |
| Fedora, openSUSE, RHEL | `combitab-*.x86_64.rpm` |

The packages are not digitally signed, so Windows and macOS show a warning on
first launch. **[docs/INSTALLATION.md](docs/INSTALLATION.md)** walks through it
step by step — no technical background required.

## Documentation

| Document | What it covers |
|---|---|
| [**Quick Start Guide**](docs/QUICKSTART.md) | Getting CombiTab, loading data, the five views, a first analysis. Ten minutes. |
| [**Complete Guide**](docs/GUIDE.md) | The full reference, including the mathematics behind the seriation methods, quality metrics, correspondence analysis and the bootstrap, plus exact export formats. |
| [**Installation**](docs/INSTALLATION.md) | Step-by-step installation per platform, security warnings, uninstalling *(German)*. |
| [**Build**](docs/BUILD.md) | Building packages, release process, code signing *(German)*. |

---

## Principles

- **Everything stays local.** There is no backend. Data is never uploaded; the
  app is installable and works offline after the first visit. The desktop
  edition goes further: a content-security policy forbids any outbound
  connection there, and even the fonts are bundled rather than loaded from a
  CDN.
- **No third-party libraries for the domain logic.** SVD, correspondence
  analysis, PDF generation, internationalisation, colour metrics and icon
  generation are written in-house. The only runtime dependencies are React and
  — loaded only if XLSX is actually used — SheetJS.
- **Limits are named, not glossed over.** Where a budget is missed, where a
  figure is a target rather than a measured result, and where the tool
  deliberately does *not* make a claim, the text says so — see "Honest limits".

## Building from source

```bash
npm ci
npm run dev        # development server (Vite)
npm run build      # type-check + production build into dist/
npm run preview    # view the build locally
npm test           # the full test suite
npm run electron   # build and launch inside the desktop shell
npm run smoke      # self-test of the desktop edition
npm run dist       # installer packages for the current system
npm run bench      # performance benchmark
npm run validate   # the CA reference validation only
npm run gen-icons  # regenerate icons (PWA + package icon)
```

Details on packaging, releasing and code signing:
**[docs/BUILD.md](docs/BUILD.md)**.

A demo dataset is loaded at startup. Load your own files by **drag and drop** or
via "Load file…": a v1 project (`.json`, migrated automatically), a v2 project
(`.json`), or a raw table (`.csv`/`.tsv`/`.xlsx`).

---

## Features

### Matrix

- **WebGL2 rendering** of the combination table, driven by the `ProjectV2`
  model: cell colour = the material-group colour of the column, saturation by
  frequency.
- **Zoom** (mouse wheel, around the cursor) and **pan** (dragging the surface),
  operable by mouse and by touch alike (pointer events).
- **Hover inspector** and **click selection** of cell / context / type with
  cross-highlighting.
- **Drag-reorder:** dragging a row or column label reorders contexts or types by
  hand. Reordering is **O(1)** — only a small order-lookup texture is re-uploaded,
  not the cells.
- **Fixed elements:** pin contexts and types in the Inspector. Pinned elements
  keep their position; seriation arranges only the free elements into the gaps.
  Mirrored in `rowMetadata.isFixed` / `columnMetadata.isFixed`.
- **Undo/redo** via snapshots of order + pinning (Ctrl/⌘+Z, Ctrl/⌘+Shift+Z and
  Ctrl+Y).
- **Overview minimap** with a viewport rectangle for fast navigation in large
  matrices.
- **Canvas-2D fallback path:** without WebGL2, the same renderer draws through a
  2D context behind the same interface — with an exactly reproduced colour model
  (`matrix/cellColor.ts`). `renderer.backend` reports
  `"webgl2" | "canvas2d" | "none"`.

### Seriation & analysis

- **Three methods:** the centroid method (reciprocal averaging), CA-based
  seriation and iterative optimisation — with a live quality score
  (concentration, continuity, anti-Robinson, total).
- **Correspondence analysis:** exact, SVD-based CA (`analysis/ca.ts`, one-sided
  Jacobi in `analysis/svd.ts`) with a **biplot** (dim 1 × 2) and a **scree plot**.
- **Ford diagram:** "battleship curves" — bar width = the type's share of the
  context's inventory, contexts in the current seriation order.
- **Bootstrap stability:** multinomial resampling of the type frequencies per
  context; CA dimension 1 per replicate, aligned to the sign of the reference.
  **Caterpillar plot** with a 90 % rank interval, median and reference rank;
  global stability as the mean |Spearman ρ|. 100/200/500 replicates,
  reproducible via a mulberry32 seed.
- **Brushing & linking:** one shared selection/hover layer (`link.tsx`) couples
  matrix, biplot and Ford; the selection survives a tab change.

### Filters & focus

Material-group chips, range filters over rows/columns, "hide empty
rows/columns", and a **focus mode** on the neighbourhood of the selection. The
filtered view is consumed identically by every view; reordering, seriation,
pinning and annotation act on the visible subset and are written back into the
base project. Active filters are carried into the share link, autosave and
project export.

### Annotations & metadata

- **Cell annotations** ("Annotate" mode): drag a cell range and set it as a
  batch — certainty (certain/uncertain/questionable), fragmentation, count
  range, inventory numbers, notes. Annotated cells carry a traffic-light marker.
  When applying to a mixed selection, **only fields actually touched** are set;
  inconsistent fields are left intact. Explicitly clearing a field still deletes
  it.
- **"Not recorded" vs. structural absence:** a cell can be marked as a missing
  value, cleanly separated from a true 0. The matrix draws such cells with a
  diagonal cross; the Inspector shows a presence statistic (present / absent /
  not recorded). Import recognises `?`, `NA` and `N/A`; export writes `?` back.
- **Metadata tab:** material-group management (see above) and a searchable type
  table for assigning groups and marking lead types.

### Colour metrics & accessibility

- **CVD-safe palette** at the press of a button (Okabe-Ito), previews under
  deuteranopia/protanopia/tritanopia and a warning about barely distinguishable
  colour pairs (perceptual distance ΔE below threshold). Colour core
  `core/palette.ts`: WCAG contrast, dichromacy simulation (Viénot 1999, correctly
  in linear light), CIELAB ΔE76.
- **Keyboard & screen reader:** skip link, `:focus-visible` focus ring, tabs
  following the WAI-ARIA tabs pattern (roving `tabindex`, arrow keys/Home/End),
  landmarks, `role="img"` with a descriptive label on the matrix canvas,
  `role="alert"` for errors, `<html lang>` following the UI language.
- **`prefers-reduced-motion`** disables non-essential transitions.

### Export

Via the export menu in the header, reachable from every tab:

- **Matrix image** in the current display order: **SVG** and **PDF** (true
  vector, no third-party library — PDF written directly as a content stream with
  base-14 Helvetica) as well as **PNG** (rasterised ×2). Pinned elements are
  labelled in red, material groups appear as a legend.
- **Sorted data:** CSV and XLSX in seriation order.
- **Project file:** v2 (canonical schema, `.combitab.json`) and, via the
  migration adapter, v1 (backward-compatible).
- **Methods report** as Markdown, **bilingual**, with dataset figures, method,
  quality metrics, CA inertia, bootstrap stability and citation.
- **Linked Open Data** as Turtle and JSON-LD (see below).

Hardened cross-browser: CSV with a UTF-8 BOM for Excel on Windows, SVG with XML
prolog and namespace, PDF with `/MediaBox`, transliterated portable filenames,
and a `window.open` fallback for browsers without `<a download>`.

### Linked Open Data (CIDOC-CRM / CRMarchaeo)

- Contexts are modelled as `crmarchaeo:A2_Stratigraphic_Volume_Unit` (not as
  sites); the **single** `crm:E27_Site` is the findspot. Each context gets a
  `crmarchaeo:A1_Excavation_Processing_Unit` with
  `crmarchaeo:AP5_removed_part_or_all_of`.
- **Real object identities:** a cell with frequency n produces n individual
  `crm:E22_Human-Made_Object`, located in the context via
  `crmarchaeo:AP21_contains`. A documented limit (`maxObjectsPerCell`, default
  1000) falls back explicitly to an aggregate above that; `objectIdentities:
  false` forces the aggregate.
- **A shared intermediate representation:** Turtle and JSON-LD are rendered from
  *one* IR graph and are guaranteed to carry the same statements.
- "Not recorded" **and** structural absence deliberately produce **no** find.

### Sharing, autosave, offline

- **Share link:** the complete state (project, order, filters, annotations,
  missing-value marks, active tab) is encoded into the **URL fragment** via the
  native `CompressionStream` and base64url — the fragment never goes to a
  server. Above ~16 000 characters, a notice suggests sharing the project file
  instead.
- **Autosave in IndexedDB** when the page is hidden or closed; on the next start
  a banner offers to restore. Every access is fault-tolerant (private mode →
  silently disabled).
- **PWA:** manifest, service worker (app shell cached on install, navigations
  network-first with an offline fallback, other requests
  stale-while-revalidate), install button. Icons are generated dependency-free
  via Node's `zlib`.

### Bilingual

The entire visible interface exists in German and English, including the canvas
labels. Switch with the DE/EN button; the choice is remembered and pre-set from
the browser language at startup. An in-house i18n core (`src/i18n/`) with a flat
dictionary, `{var}` interpolation and an EN → DE → key fallback.

---

## Desktop edition

The same application, packaged in an Electron shell — so that users need
neither Node.js, nor a terminal, nor a package manager.

- **It loads through its own `app://` scheme, not `file://`.** That is not a
  detail: the compute workers are **module workers**
  (`new Worker(url, { type: "module" })`), and Chromium refuses to load those
  over `file://` — correspondence analysis, bootstrap and score computation
  would have failed silently. The custom scheme also supplies a stable origin,
  and with it reliable IndexedDB (autosave), localStorage (theme, language) and
  a secure context for `CompressionStream` (share links).
- **Locked down.** `contextIsolation`, `sandbox`, no `nodeIntegration`. The
  preload script passes through **no** file or Node function, only an identifier
  by which the interface recognises the shell. A content-security policy forbids
  every outbound connection.
- **Exports land where you want them:** in the shell, the app's `<a download>`
  path opens a native save dialog instead of silently writing to the downloads
  folder.
- **A self-test instead of an assumption.** `npm run smoke` starts an invisible
  window and checks the ten properties that can break when an app moves into a
  shell — module workers, WebGL2, secure context, loaded fonts, no external
  resources. It runs in CI and has also been run against the finished AppImage.

---

# CombiTab 2 — Deutsch

[English ↑](#combitab-2)

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
| Linux, universell | `CombiTab-*-x86_64.AppImage` |
| Ubuntu 22.04+, Debian 12+, Mint | `combitab_*_amd64.deb` |
| Fedora, openSUSE, RHEL | `combitab-*.x86_64.rpm` |

Die Pakete sind nicht digital signiert; Windows und macOS zeigen deshalb beim
ersten Start eine Warnung. **[docs/INSTALLATION.md](docs/INSTALLATION.md)**
führt Schritt für Schritt hindurch — auch ohne technische Vorkenntnisse.

## Dokumentation

| Dokument | Inhalt |
|---|---|
| [**Quick Start Guide**](docs/QUICKSTART.md) | CombiTab beziehen, Daten laden, die fünf Ansichten, eine erste Auswertung. Zehn Minuten *(englisch)*. |
| [**Complete Guide**](docs/GUIDE.md) | Das vollständige Nachschlagewerk, samt der Mathematik hinter Seriationsverfahren, Qualitätsmetriken, Korrespondenzanalyse und Bootstrap sowie den exakten Exportformaten *(englisch)*. |
| [**Installation**](docs/INSTALLATION.md) | Installation Schritt für Schritt je System, Sicherheitswarnungen, Deinstallation. |
| [**Build**](docs/BUILD.md) | Paketbau, Freigabe, Code-Signierung. |

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

