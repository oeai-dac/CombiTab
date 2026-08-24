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
| Linux, universell | `CombiTab-*-x86_64.AppImage` |
| Ubuntu 22.04+, Debian 12+, Mint | `combitab_*_amd64.deb` |
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

