# CombiTab 2 — Complete Guide

*Archaeological seriation and combination-table analysis*

**Version:** 2.0.0 · **License:** MIT · **Author:** Christian Gugl, Austrian
Archaeological Institute, Austrian Academy of Sciences (ÖAW)

**Written with:** Anthropic's Claude, in Claude Code. Large parts of the
application and of this documentation were developed with its assistance. The
direction of the work, the archaeological and methodological decisions, the
testing and the responsibility for the result lie with the author.

> This is the full reference. If you only want to get started, read the
> [**Quick Start Guide**](QUICKSTART.md) instead — ten minutes, no mathematics.
> Step-by-step installation instructions for people without a technical
> background are in [**INSTALLATION.md**](INSTALLATION.md) (German); packaging
> and release details are in [**BUILD.md**](BUILD.md).

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installing and Running CombiTab](#2-installing-and-running-combitab)
3. [Data Model & File Formats](#3-data-model--file-formats)
4. [User Interface Overview](#4-user-interface-overview)
5. [The Matrix View](#5-the-matrix-view)
6. [Seriation Methods — Mathematical Foundations](#6-seriation-methods--mathematical-foundations)
7. [Quality Metrics — Mathematical Foundations](#7-quality-metrics--mathematical-foundations)
8. [Correspondence Analysis View](#8-correspondence-analysis-view)
9. [The Ford Diagram](#9-the-ford-diagram)
10. [The Metadata View — Material Groups & Type Assignment](#10-the-metadata-view--material-groups--type-assignment)
11. [Cell Annotations](#11-cell-annotations)
12. [Missing Data vs. Structural Absence](#12-missing-data-vs-structural-absence)
13. [Stability Analysis — Mathematical Foundations](#13-stability-analysis--mathematical-foundations)
14. [Project-Level Metadata](#14-project-level-metadata)
15. [Export & Interoperability](#15-export--interoperability) (incl. exact formats & the stratigraphy limitation)
16. [Sharing, Autosave & Offline Use](#16-sharing-autosave--offline-use)
17. [Performance & Large Datasets](#17-performance--large-datasets)
18. [Accessibility](#18-accessibility)
19. [Keyboard Shortcuts](#19-keyboard-shortcuts)
20. [End-to-End Workflows](#20-end-to-end-workflows)
21. [Architecture Notes](#21-architecture-notes)
22. [Troubleshooting](#22-troubleshooting)

---

## 1. Introduction

**Seriation** is one of the oldest quantitative methods in archaeology: given a
set of archaeological contexts (most often graves) and the artifact types
found in each, seriation seeks a single ordering of contexts — and,
simultaneously, of types — such that the resulting occurrence matrix shows a
smooth, unimodal pattern of artifact popularity over time (the classic
"battleship curve"). A good seriation is a strong proxy for **relative
chronology**: contexts placed close together in the ordering are assumed to
be close together in time, because they share similar assemblages.

CombiTab is a modern, browser-based implementation of this method, built
around a **combination table** (Kombinationstabelle in the German
archaeological tradition): a matrix of contexts × types, either as raw
frequencies or as presence/absence data. CombiTab lets you:

- Import raw tabular data or existing project files.
- Automatically reorder (seriate) the matrix using three different
  mathematical methods.
- Manually fine-tune the ordering by dragging rows/columns and pinning
  well-dated "anchor" contexts.
- Quantify how good an ordering is with transparent, documented quality
  metrics.
- Assess the statistical robustness of the resulting chronology via
  bootstrap resampling.
- Annotate individual finds with certainty, fragmentation, and inventory
  data.
- Export publication-ready graphics, re-usable data, and semantic web /
  Linked-Open-Data representations (CIDOC-CRM / CRMarchaeo).

The application is fully client-side (no server, no database), built with
React, TypeScript and a hand-written WebGL2 renderer. It ships in two forms
from the same source: a **desktop application** for Windows, macOS and Linux,
and a **web version** that installs as an offline-capable Progressive Web App.
The feature set is identical; §2 explains which to choose. The interface is
available in German and English throughout.

---

## 2. Installing and Running CombiTab

CombiTab ships in two forms built from the same source: a **desktop
application** for Windows, macOS and Linux, and a **web version** that runs in
the browser and stays usable offline. Which one you want depends on the
situation, not on what the software can do — the feature set is identical.

### 2.1 The web version — nothing to install

**<https://oeai-dac.github.io/CombiTab/>**

All computation happens in your own browser; no data is uploaded, because there
is no server to upload it to. This is the fastest way to try the software, the
only route on a managed machine where you cannot install anything, and the
recommended route on Intel Macs, for which no package is built.

Requirements: a current browser with **WebGL2** — Chrome, Edge and Firefox all
qualify. Without WebGL2 the matrix falls back to a Canvas-2D renderer that
produces identical output more slowly (§18). Firefox has proven the most
reliable browser for triggering file *downloads*; some Chromium configurations
and extensions interfere with download prompts.

### 2.2 The desktop application

Download the file for your system from the
[Releases](https://github.com/oeai-dac/CombiTab/releases) page:

| System | File |
|---|---|
| Windows 10 / 11 | `CombiTab-2.0.0-Setup-x64.exe` |
| Windows, without installing | `CombiTab-2.0.0-portable-x64.exe` |
| macOS, Apple silicon (M1–M4) | `CombiTab-2.0.0-arm64.dmg` |
| Linux, any distribution | `CombiTab-2.0.0-x86_64.AppImage` |
| Ubuntu 22.04+, Debian 12+, Mint | `combitab_2.0.0_amd64.deb` |
| Fedora, openSUSE, RHEL | `combitab-2.0.0.x86_64.rpm` |

The packages are **not code-signed**, so Windows and macOS show a warning on
first launch: on Windows, "More info" → "Run anyway"; on macOS, right-click the
app and choose "Open" rather than double-clicking. This is a statement about
publisher registration, not about the software. The Windows installer needs no
administrator rights — it installs into the user profile.
[INSTALLATION.md](INSTALLATION.md) covers each platform step by step in German,
including the two Ubuntu 24.04 quirks that affect every AppImage (missing FUSE 2
and restricted user namespaces) and the `xattr -cr` remedy for macOS's
misleading "is damaged" message.

Three things distinguish the desktop edition technically:

- It loads through a **custom `app://` scheme rather than `file://`**. This is
  not cosmetic. The compute workers are ES **module workers**, which Chromium
  refuses to load over `file://` — correspondence analysis, bootstrap and score
  computation would fail silently. The custom scheme also supplies a stable
  origin, and therefore working IndexedDB (autosave), localStorage (theme,
  language) and a secure context for `CompressionStream` (share links).
- It is **locked down**: context isolation, sandbox, no Node integration, and a
  content-security policy that forbids any outbound connection. The preload
  script exposes no file or Node API — only a flag by which the interface
  recognises that it is running in the shell. Even the fonts are bundled rather
  than fetched from a CDN.
- **Exports open a native save dialog** instead of dropping files silently into
  a downloads folder.

The **Help** menu opens the project page or the issue tracker in your browser;
that is the only outbound action in the whole application, and it needs an
explicit click.

### 2.3 Building from source

Requirements: **Node.js 18 or later** (developed and tested on Node 22) and npm.

```bash
npm ci             # exact dependency install from package-lock.json
npm run dev        # Vite dev server with hot reload, typically :5173
npm run build      # tsc type-check, then a production bundle into dist/
npm run preview    # serve the production build locally
npm run electron   # build and launch inside the desktop shell
npm run smoke      # self-test of the desktop edition (10 checks)
npm run dist       # installer packages for the current platform
```

Runtime dependencies are deliberately minimal: React, and SheetJS (`xlsx`)
loaded only when XLSX import or export is actually used. Correspondence
analysis, SVD, PDF generation, internationalisation, colour metrics and icon
generation are all written in-house rather than pulled from libraries.

`dist/` is a fully static site — HTML, JS, CSS and a service worker — deployable
to any static host (nginx, Apache, GitHub Pages, S3, `python -m http.server`).
There is no backend, database or server-side runtime.

**Cross-platform packaging has a hard limit:** Windows and macOS packages cannot
be built on Linux (NSIS needs Wine; `sips`, `hdiutil` and `codesign` exist only
on macOS). Both are therefore produced by the release workflow on real runners,
which also installs each finished package and launches it before a release draft
is created. Details in [BUILD.md](BUILD.md).

### 2.4 Running the test suite

```bash
npm test           # the full suite
npm run bench      # in-process performance benchmark
npm run validate   # the CA reference validation only
```

The suite covers the seriation algorithms, the quality metrics, correspondence
analysis validated against Greenacre's published reference values, import/export
round-trips, i18n completeness, filter logic, missing-value semantics,
annotation semantics, share-link encoding and cross-browser export behaviour.

### 2.5 Installing the web version as an offline app (PWA)

Opened in Chrome or Edge, CombiTab offers an **install** button (also available
in the browser's address bar). After installation:

- The app works **fully offline**. A service worker caches the app shell and
  bundled assets, serving navigations network-first with an offline fallback and
  other requests stale-while-revalidate.
- It runs in its own window, without browser chrome.
- **Touch input is fully supported**: the matrix responds to pointer events for
  panning, drag-reordering rows and columns, and rectangular area selection —
  which makes a tablet a practical device in the field or in a storeroom. Zoom
  on touch is via the cell-size slider rather than pinch gestures.

---

## 3. Data Model & File Formats

### 3.1 The project model (`ProjectV2`)

Internally, every open dataset is a single JSON-serializable object
containing:

- `contexts` / `types` — ordered name arrays (the canonical row/column
  identity; all internal indices refer back to these by position).
- `matrix` — a dense 2D array of numbers (`contexts.length × types.length`),
  either raw frequencies or 0/1 presence values, controlled by `dataType`.
- `columnMetadata` / `rowMetadata` — per-type and per-context metadata:
  material group, color, "index type" flag, fixed/pinned flag, free-text
  notes, context type, area.
- `cellAnnotations` — a sparse map keyed by canonical `row:col`, holding
  certainty, fragmentation, count range, inventory numbers, and notes per
  cell.
- `missingCells` — a sparse set of canonical cells explicitly marked "not
  recorded" (distinct from a true zero — see §12).
- `order` — the current **display order** (row/column name arrays in display
  sequence), independent of the canonical index order used internally.
- `materialGroups` — a map of material-group name → color.
- `filters` — the currently active filter/focus settings (material filter,
  row/column range, hide-empty toggles); persisted with the project so
  filtered views survive a reload, a share link, or an export.
- `view` — display preferences (cell size, which overlays are shown).
- `history` — an append-only, human-readable log of structural operations
  (seriation runs with their method and seed, imports, migrations) used to
  auto-generate a reproducible methods paragraph (§15).

This separation between **canonical indices** (stable identity, used for
annotations and metadata) and **display order** (what you see on screen,
changed constantly by dragging and seriation) is a core design principle: it
lets annotations, colors, and fixed/pinned state survive any amount of
reordering, because they're keyed by name/canonical position, never by
on-screen position.

### 3.2 Supported import formats

| Format | Notes |
|---|---|
| `.csv` / `.tsv` | A raw context × type table, in either **wide** or **long** layout (§3.2.1–3.2.2). Delimiter is auto-detected. |
| `.xlsx` | Same interpretation, read from the first worksheet of the workbook via SheetJS, applying the identical wide/long parsing logic used for CSV. Loaded on demand (see §15.2). |
| `.json` (CombiTab v2 project) | The full native project format — matrix, metadata, annotations, missing-cell markers, view state, filters, and history. Round-trips losslessly. |
| `.json` (CombiTab v1 project) | The legacy project format from the first CombiTab generation. Automatically detected (by schema shape) and migrated to v2 on import — `snake_case` fields such as `filter_row_range` are converted to their `camelCase` v2 equivalents (`rowRange`), and the migration is logged in `history`. Re-importing an already-migrated file is idempotent. |

Files can be loaded via **drag-and-drop** anywhere in the application window,
or via the **"Load File…"** control.

#### 3.2.1 Wide format (default)

The standard combination-table layout: one row per context, one column per
type, first row and first column reserved for names.

```csv
Context,Fibula A,Bead,Spindle Whorl
Grave_1,3,0,1
Grave_2,0,2,?
Grave_3,1,1,0
```

- **First row** = type names (the header). Can be disabled with
  `hasHeader: false`, in which case columns are auto-named `Typ_1`, `Typ_2`, …
- **First column** = context names (the index column). Can be disabled with
  `hasIndexColumn: false`, in which case contexts are auto-named
  `Kontext_1`, `Kontext_2`, …
- `transpose: true` swaps rows and columns *before* parsing, for source
  tables where types run down the rows instead.
- Cell values may be integers, decimals (both `3.5` and the European `3,5`
  are accepted), `0`/`1` presence flags, or a missing-value token (below).
- Context and type names must be **unique** within the file — a duplicate
  name raises an import error rather than silently merging or overwriting
  rows.

#### 3.2.2 Long format

One row per (context, type, count) observation — the layout typical of a
finds-database export or an SQL query result:

```csv
context,type,count
Grave_1,Fibula A,3
Grave_1,Spindle Whorl,1
Grave_2,Bead,2
```

- Enabled via `format: "long"`; column identification defaults to
  positions 0/1/2 but can be overridden by name or index:
  `long: { context: "grave_id", type: "artifact_type", count: "qty" }`.
- Contexts and types are discovered in first-seen order and assembled into
  a dense matrix.
- **Duplicate (context, type) pairs are summed**, not overwritten — e.g. two
  rows both giving `Grave_1, Fibula A` add their counts together. The
  import report notes how many duplicate pairs were merged this way.
- A row with an empty context or type is skipped, with a warning citing the
  line number.

#### 3.2.3 Missing-value tokens

On import, the following tokens are recognized case-insensitively in any
cell and imported as **"not recorded"** (§12) rather than as a numeric
zero: **`?`**, **`NA`**, **`N/A`**. Any other non-numeric, non-empty cell
value is treated as a data error: it is imported as `0`, and a warning
citing the context, type, and offending value is added to the import
report (capped at the first 12 such warnings to avoid flooding the report
on badly malformed files).

#### 3.2.4 Automatic data-type detection

Unless `dataType` is explicitly set, CombiTab inspects the parsed matrix:
if every non-zero cell equals exactly `1`, the project is tagged
`presence_absence`; otherwise it's tagged `frequency`. The detection is a
property of the data, so it is almost always right; where it is not — a
frequency dataset that happens to contain only single finds — the tag can be
corrected in the project JSON (§14), which does not alter the stored values.

#### 3.2.5 What a fresh import looks like

A freshly imported table is assembled into a full `ProjectV2` (§3.1) with
sensible defaults: all types start in an `"Unassigned"` material group
(mid-grey `#808080`), a starter palette of common archaeological material
groups is pre-populated (`Ceramic`, `Metal`, `Glass`, `Bone/Antler`,
`Stone`, `Organic`, each with a distinct default color) even though nothing
is assigned to them yet, a starter list of common context types is
pre-populated (`Grave`, `Pit`, `Ditch`, `Layer`, `Posthole`, `Well`), the
display order (`order.rows` / `order.cols`) is initialized to the import
order, and `history` is empty — ready for you to assign materials (§10) and
run a first seriation (§6).

### 3.3 Data type: frequency vs. presence/absence

A project is tagged as either:

- **Frequency data** — cell values are counts (e.g. number of sherds of a
  given type in a grave). Cell color saturation and the quality metrics use
  the actual magnitude.
- **Presence/absence data** — cell values are 0 or 1. All the same algorithms
  apply; frequency-sensitive weighting simply degenerates to binary
  weighting.

The tag is set automatically at import (§3.2.4) and travels in the project file
(§14). It does not alter the stored matrix values, only how they are interpreted
for weighting.

---

## 4. User Interface Overview

The window has three parts.

**The header** carries the application title and subtitle, a light/dark theme
toggle, the **DE/EN** language switch, **Load file…**, **Share**, the **Export**
menu, and — in the web version — an **Install app** button once the browser
offers installation.

**The tab bar** switches between **five** views:

| Tab | What it is | Section |
|---|---|---|
| **Matrix** | The seriation workspace: the combination table itself, the seriation controls, the quality score, the filters | §5 |
| **Correspondence analysis** | Biplot on freely selectable axes, plus a scree plot of explained inertia | §8 |
| **Ford diagram** | Battleship curves in the current seriation order | §9 |
| **Stability** | Bootstrap robustness of the ordering, as a caterpillar plot | §13 |
| **Metadata** | Material groups, the colourblind-safe palette, and the type table | §10 |

Two things people expect to find as tabs are not tabs, because they belong to
the matrix itself: **seriation** is run from the matrix toolbar (§6), and
**annotation** is a *mode* of the matrix rather than a separate editor (§11).

**The Inspector**, on the right, shows contextual detail for whatever is
hovered or selected — a cell, a context or a type: value, material group,
presence statistics, project metadata such as context type, area and notes, the
pin/unpin control, and the annotation editor when the matrix is in annotate
mode.

All four analytical views share **one selection and hover state**. Pointing at a
context in the Ford diagram highlights it in the matrix, in the CA plot and in
the stability chart at the same time, and the selection survives a tab switch —
so an outlier can be traced through every representation without losing your
place.

The interface follows the ÖAI (Austrian Archaeological Institute) visual style:
a warm off-white ground, a deep red accent used for headings and the "fixed"
indicator, and Outfit as the sans-serif face. Light and dark themes are both
available. The entire interface, including the labels drawn onto the canvases,
exists in German and English, held in a small in-house translation dictionary
with an automated test that no key is missing or empty in either language. The
initial language follows the browser or system setting; your choice is
remembered.

---

## 5. The Matrix View

The Matrix View is a custom **WebGL2-rendered** canvas (with an automatic
Canvas-2D fallback for systems without WebGL2, so the app remains usable —
with reduced performance on very large matrices — even on older hardware).

### 5.1 Reading the matrix

- **Rows** = contexts (graves), in the current display order.
- **Columns** = types (artifact categories), in the current display order.
- **Cell color** = the material group assigned to that column (set in the
  Metadata view, §10); **saturation** = the cell's value, so heavily represented
  types in a context stand out visually.
- Row and column labels are drawn along the left and top margins; the
  corner shows small axis titles ("Context ▾", "Type ▸").
- A **legend** below the matrix lists all material groups with their color,
  a marker explaining the "fixed" indicator, and a live count of existing
  cell annotations.

### 5.2 Navigation

- **Pan**: click-and-drag (or touch-drag) on any empty area of the canvas.
- **Zoom**: mouse wheel, zooming around the cursor position; or the
  cell-size slider in the toolbar. On touch devices, use the slider (no
  pinch gesture is implemented).
- **Minimap**: a small overview panel renders a downsampled density map of
  the entire matrix (material-colored, in display order) with a rectangle
  showing your current viewport. Click or drag on the minimap to jump the
  main view to a different region — essential for navigating matrices with
  hundreds of rows/columns.
- **Hover** shows a live tooltip and updates the Inspector panel; **click**
  selects a cell, row, or column, with the corresponding row/column
  cross-highlighted.

### 5.3 Reordering

- **Drag-reorder**: grab a row or column label and drag it to a new
  position. This works identically whether the reordering was triggered
  manually or you're nudging the result of an automatic seriation run.
  Internally, reordering is **O(1)** — only a small order-lookup texture is
  re-uploaded to the GPU, not the full cell grid — so dragging stays smooth
  even on matrices with 100+ rows and columns.
- **Fixed / pinned elements**: any context or type can be pinned in place
  (via the Inspector panel's "pin" toggle, or the pin icon shown on hover).
  A pinned element:
  - Cannot be dragged.
  - Keeps its exact display position when an automatic seriation method is
    run — the algorithm only reorders the *free* (unpinned) rows/columns and
    slots them into the remaining positions in the order the algorithm
    produced, leaving pinned rows/columns untouched.
  - Is visually marked with a small accent-colored dot next to its label.

  This is the standard way to anchor a well-dated context (e.g. a grave with
  independent absolute dating, or a coin) so that automatic seriation
  respects it as a fixed chronological reference point.

### 5.4 Interaction modes

The toolbar's **Mode** control switches the matrix between two behaviours:

- **Navigate** (default): click selects a single cell, context or type;
  dragging the canvas pans the view, and dragging a *label* reorders.
- **Annotate**: dragging on the cell grid instead draws a rectangular **area
  selection**, used to batch-annotate cells or batch-mark them as not
  recorded (§11, §12).

Zooming with the wheel, the cell-size slider and the Fit button work in both
modes.

### 5.5 Focus mode

**Focus on selection** is a toggle in the filter panel, available once a
context or a type is selected (until then it tells you to select one first).
It narrows the visible matrix to the selection's direct neighbourhood: for a
focused context, the types occurring in it — and then every context carrying
one of those types, and every type occurring in those contexts. For a focused
type, the symmetric expansion. This is the fast way to answer "what else
occurs together with X?" without building a filter by hand.

Focus composes with the other filters rather than replacing them, and it is
part of the state a share link carries.

### 5.6 Filters

The filter panel lets you restrict the visible matrix by:

- **Material group** — show only types belonging to selected material
  groups.
- **Row range** / **column range** — restrict to a contiguous band of
  canonical indices.
- **Hide empty rows** / **hide empty columns** — after any other filter is
  applied, remove rows/columns that are entirely empty within the remaining
  visible set (computed consistently: columns are pruned first, then rows,
  against the already-pruned columns).

Filtering produces a genuine, self-contained **filtered project view**
(same shape as a full project) that all linked views (Matrix, CA, Stability)
consume identically — annotations and missing-cell markers are re-keyed by
stable context/type name onto the new, smaller set of canonical indices, so
nothing is lost or misaligned. Active filters are saved as part of the
project, so they're preserved across reloads and are included when you
generate a share link or export the project file.

### 5.7 Performance HUD

Press **Shift+P** (ignored while typing in a text field) to toggle a small
overlay reporting live frames-per-second, average and 95th-percentile frame
and draw times, the rendering backend (WebGL2 or Canvas-2D fallback), and
total vs. currently visible cell counts. This is primarily a diagnostic tool
for reporting performance issues on very large datasets.

---

## 6. Seriation Methods — Mathematical Foundations

CombiTab offers three seriation algorithms, selectable from the Seriation
control in the Matrix toolbar. All three produce a complete display order
for both rows and columns; all three are **fix-aware** by construction —
pinned elements retain their position, and the algorithm's output order is
used only to fill the remaining free slots (see §5.3).

### 6.1 Centroid method (reciprocal averaging)

This is the classical archaeological seriation technique, closely related to
**correspondence analysis** and to the "reciprocal averaging" algorithm used
in ecology and quantitative archaeology since the 1970s.

**Algorithm.** Let $M$ be the $N_R \times N_C$ occurrence matrix. Row weights
$w_i = \sum_j M_{ij}$ and column weights $w_j = \sum_i M_{ij}$ are computed
once. A row score vector $r \in \mathbb{R}^{N_R}$ is initialized from a
seeded pseudo-random generator (so runs are exactly reproducible given the
same seed — the seed is recorded in the project's `history` log for
provenance). The method then alternates, for a fixed number of iterations
(default 15):

1. **Column update** — each column's score is the weighted average of the
   row scores of the rows it occurs in:
   $$c_j = \frac{\sum_i M_{ij}\, r_i}{\sum_i M_{ij}}$$
2. **Row update** — symmetrically, each row's score becomes the weighted
   average of the column scores:
   $$r_i = \frac{\sum_j M_{ij}\, c_j}{\sum_j M_{ij}}$$
3. **Rescaling** — after each row update, $r$ is linearly rescaled to
   $[0, 1]$ to prevent numerical drift over iterations.

This is the power-iteration method for finding the dominant non-trivial
eigenvector of the row/column co-occurrence structure — in the limit, it
converges toward the same ordering that the first axis of a correspondence
analysis would produce, but is computed directly without an SVD, making it
fast even on large matrices.

Cells marked "not recorded" (§12) are excluded from both the weights and the
weighted sums, so missing data does not silently act as a zero.

Finally, both the row and column score vectors are **argsorted** (sorted
ascending, keeping track of original indices) to produce the final display
order.

### 6.2 Correspondence Analysis (CA) seriation

Rather than power iteration, this method computes an exact **correspondence
analysis** of the matrix (full mathematical treatment in §8) and orders
contexts and types by their coordinate on a chosen CA axis (by default, the
first axis, which captures the greatest share of the table's total inertia
— archaeologically, this is very often the chronological "horseshoe" axis in
well-behaved seriation data).

Rows and columns are seriated independently but consistently: row scores
come from the CA of the matrix itself; column scores come from the CA of the
transposed matrix (with the missing-value mask transposed correspondingly),
guaranteeing the same treatment of missing cells on both axes.

### 6.3 Iterative optimization (hill-climbing refinement)

This method starts from the **centroid method's** output and then refines it
by **greedy neighbor-swapping**: for each adjacent pair of rows (and,
separately, columns) in the current order, the algorithm computes the change
in weighted diagonal distance (the core term of the concentration metric,
§7.1) that swapping them would produce, and **accepts the swap only if it
strictly decreases** that distance. This is repeated in sweeps (default up
to 30 passes) until no improving swap is found or the pass budget is
exhausted.

Each swap's cost is evaluated incrementally in $O(N_C)$ (for a row swap) or
$O(N_R)$ (for a column swap) — not by recomputing the whole matrix's score —
which keeps the method tractable even on matrices with hundreds of rows and
columns.

Because only strictly improving swaps are accepted, this method is
**guaranteed to reach a concentration score at least as good as** the plain
centroid method it starts from — it can only improve the diagonal
concentration, never worsen it. It typically converges to a local optimum
near the centroid solution rather than a global one, which is why it's
recommended as a **polishing step** after an initial centroid or CA pass,
not as the first method to try on an unordered matrix (see §17).

### 6.4 Choosing a method

| Method | Best for | Determinism |
|---|---|---|
| Centroid | First pass on any dataset; fast, well-tested classical approach | Deterministic given the seed |
| CA | Cross-checking the centroid result; data with a strong, well-separated chronological signal | Deterministic (exact eigendecomposition) |
| Iterative | Polishing an already-reasonable order; squeezing out local improvements | Deterministic given the seed of its centroid starting point |

All three methods record their name, and — for the auto-generated methods
report (§15) — a citation-ready English description of the technique used
("reciprocal averaging", "correspondence analysis seriation (CA dim. 1)",
"iterative seriation (concentration optimization)").

---

## 7. Quality Metrics — Mathematical Foundations

After every seriation run (or manual reorder), CombiTab computes a live
**quality score** in $[0, 1]$ (higher is better), combining three
independent, individually meaningful metrics.

### 7.1 Concentration

Measures how tightly artifact occurrences cluster around the matrix
diagonal — the visual hallmark of a good seriation. For each non-zero,
non-missing cell $(r, c)$ with value $v$, its normalized diagonal distance is

$$d_{rc} = \left| \frac{\text{pos}(r)}{N_R-1} - \frac{\text{pos}(c)}{N_C-1} \right|$$

where $\text{pos}(\cdot)$ is the cell's position in the *current display
order* (0-indexed). The metric is the value-weighted mean distance,
inverted so that 1 = perfectly concentrated on the diagonal:

$$\text{concentration} = 1 - \frac{\sum_{r,c} v_{rc}\, d_{rc}}{\sum_{r,c} v_{rc}}$$

### 7.2 Anti-Robinson index

A matrix has the **Robinson property** if, reading along any row of a
context similarity matrix, similarity to the anchor context is
non-increasing as you move away from it in the seriation order — the
formal definition of "a good seriation."

CombiTab computes a genuine similarity matrix between *rows* (contexts)
using **cosine similarity** of their type-occurrence vectors, and then, for
every context used as an anchor, checks every pair of increasingly distant
neighbors on either side to see whether similarity is (weakly) monotonically
decreasing outward. The index is the **fraction of these neighbor
comparisons that satisfy the Robinson property**:

$$\text{antiRobinson} = \frac{\#\{\text{comparisons satisfying monotonicity}\}}{\#\{\text{total comparisons}\}}$$

A perfectly Robinson-ordered matrix scores 1. (Note: earlier CombiTab
generations used this name for a simpler gap/continuity count; the current
implementation is a true Anti-Robinson index computed on a proper
similarity matrix, documented as such in the source to avoid confusion with
that older, differently-defined metric.)

When cells are marked "not recorded", similarity between two contexts is
computed **pairwise-complete** — only over the columns where *both* contexts
have a known value — so a missing cell never silently drags similarity
toward zero the way treating it as an absence would.

### 7.3 Continuity

For each type (column), continuity measures what fraction of its
"occurrence span" (from its first to its last occurrence in the current row
order) is actually filled, rather than containing gaps:

$$\text{continuity}_j = \frac{n_j}{\text{last}_j - \text{first}_j + 1}$$

where $n_j$ is the number of contexts in which type $j$ occurs, and
$\text{first}_j$/$\text{last}_j$ are the row positions of its first and last
occurrence. The overall continuity score is the mean of this ratio across
all types that occur at least once. A type with no gaps in its occurrence
span scores 1; a type that appears, disappears, and reappears later scores
lower.

Cells marked "not recorded" that fall *inside* a type's occurrence span are
treated as genuinely unknown and excluded from the denominator, rather than
counting as a gap.

### 7.4 Combined score

$$\text{total} = 0.40 \times \text{concentration} + 0.35 \times \text{antiRobinson} + 0.25 \times \text{continuity}$$

These default weights are a documented, transparent configuration (not a
hidden black-box formula) and can be overridden programmatically for
specialized analyses. The default weighting reflects that concentration and
the Robinson property are the primary diagnostic signals of a good
chronological ordering, with continuity as a supporting, secondary
indicator.

Because the anti-Robinson index is $O(N_R^2 \times N_C)$, quality
computation for very large matrices is offloaded to a background **Web
Worker** so scoring never blocks the interface — you can keep interacting
with the matrix while a large recomputation runs, and any stale computation
is automatically cancelled if you trigger a new one before it finishes.

---

## 8. Correspondence Analysis View

The CA view provides a dedicated, interactive scatter plot of the table's
correspondence analysis solution, independent of whether you've used CA as
the seriation method.

### 8.1 Mathematical basis

Given the occurrence matrix $X$ (contexts × types) with grand total
$T = \sum_{ij} X_{ij}$:

- The **correspondence matrix** is $P = X / T$.
- **Row masses** $r_i = \sum_j P_{ij}$, **column masses**
  $c_j = \sum_i P_{ij}$.
- The **standardized residual matrix**:
  $$S_{ij} = \frac{P_{ij} - r_i c_j}{\sqrt{r_i c_j}}$$
  (this centers out the trivial "independence" structure, so no trivial
  dimension needs to be discarded afterward).
- A **singular value decomposition** $S = U \Sigma V^{\mathsf{T}}$ is
  computed.
- **Principal coordinates** (used for the biplot, so row and column points
  are on a comparable scale):
  $$F_{ik} = \frac{\sigma_k\, U_{ik}}{\sqrt{r_i}} \quad\text{(rows/contexts)}, \qquad
    G_{jk} = \frac{\sigma_k\, V_{jk}}{\sqrt{c_j}} \quad\text{(columns/types)}$$
- **Eigenvalues** $\lambda_k = \sigma_k^2$; **explained inertia** of axis
  $k$ is $\lambda_k / \sum_k \lambda_k$.

The implementation has been **validated against Greenacre's published
reference datasets and coordinate values**, matching to six decimal places
— giving confidence that results are directly comparable to CA output from
established statistical packages (e.g. R's `ca` package).

### 8.2 Missing-value handling in CA

Correspondence analysis requires a complete contingency table — individual
cells cannot be masked out the way they can in the pairwise quality metrics.
Cells marked "not recorded" are therefore filled by **iterative
proportional fitting under the independence model**: each missing cell is
repeatedly set to its expected value under row/column independence,
$E_{ij} = R_i C_j / T$, recomputed from the current (partially-filled)
margins, and iterated to convergence (a classical IPF/EM scheme).

This is a deliberately **conservative** choice: it fills unknown cells with
"no association" rather than inventing a plausible pattern via, say,
reduced-rank reconstruction, which would risk imposing structure that was
never actually observed. A context or type with *no* known data at all
correctly stays at zero — "no signal" rather than a guess.

### 8.3 Reading the scatter plot

Contexts and types are plotted on the first two CA axes (by default), with
axis labels showing the percentage of total inertia each axis explains.
Points that cluster together represent contexts (or types) with similar
co-occurrence profiles. Outlying points — a context that shares little with
any cluster, or a type that occurs almost nowhere else — are immediately
visible and worth a closer look in the Matrix or Inspector view; they often
indicate either a genuinely unusual assemblage or a data-entry issue worth
double-checking.

---

## 9. The Ford Diagram

The Ford view draws the figure most seriation publications actually print: the
**battleship curves** named after James A. Ford, who popularised the graphical
form in the 1930s and 40s.

Each row is a context, in the **current seriation order**, top to bottom. Within
a row, every type gets a horizontal bar whose width is the type's **share of
that context's inventory** — not its absolute count. Bars are centred on the
type's column axis and coloured by material group, exactly as in the matrix.

Reading it is straightforward once you know what to look for:

- A type that appears, swells and fades as you move down the diagram is
  behaving the way seriation assumes: a fashion with a beginning, a floruit and
  an end. Its outline is the "battleship".
- A type whose bars are scattered across the whole height, with gaps, is either
  chronologically insensitive (a long-lived utility form) or a sign that the
  ordering has not resolved it.
- Two types whose curves peak at clearly different heights are a chronological
  argument you can point at in a figure caption.

**Only the 30 most frequent types are drawn**, ranked by their total count
across the whole dataset and kept in the current column order. A Ford diagram
with 150 columns is unreadable on any page, so the view shows the types that
carry the chronological signal; if a specific rarer type matters to your
argument, filter the matrix down (§5.6) so that it enters the top 30 of the
filtered view.

Because the widths are **proportions rather than counts**, a context with three
finds and a context with three hundred are directly comparable — which is the
whole reason the normalisation is there. The flip side is that a context with
very few finds produces confident-looking wide bars on almost no evidence, so
read the diagram together with the per-context totals and with the stability
analysis (§13), which is precisely the check for how much weight a thinly
furnished context can carry.

The Ford view participates in the shared selection layer: hovering a context or
type here highlights it in the matrix and in the CA plot, and vice versa, so
you can trace a single outlier across all three representations without losing
your place.

---

## 10. The Metadata View — Material Groups & Type Assignment

The **Metadata** tab is where you curate the classification of your artefact
types, kept deliberately separate from the raw occurrence data. It has two
halves.

**Material groups.** Create, rename, recolour and delete named groups such as
"Ceramics", "Bronze fibulae" or "Glass beads". A group's colour is not
decoration: it drives the cell shading in the matrix, the matrix legend and the
material filter chips, so at a glance you can see which material categories
dominate which part of the seriation. Changing a group's colour recolours every
type assigned to it. Deleting a group moves its types to another group and asks
for confirmation first; the last remaining group cannot be deleted, since every
type must belong to one.

**Colour-vision safety** sits in the same panel and is worth using deliberately:

- One button applies the built-in **colourblind-safe palette** (Okabe-Ito) to
  all groups at once.
- A preview selector re-renders every group swatch as it appears under
  **deuteranopia**, **protanopia** or **tritanopia**, simulated in linear light
  after Viénot et al. (1999) rather than by naive channel-swapping.
- CombiTab then names the specific **pairs of groups that are hard to tell
  apart**, measured as a CIELAB ΔE below the discrimination threshold, instead
  of leaving you to squint at the preview. If nothing is flagged, it says so.

**The type table.** A searchable list of every type with two editable columns:
its **material group**, and an **index type** (lead type) flag marking a type as
chronologically diagnostic. The flag is carried into the Inspector, the exported
project file and the methods report.

Assignments take effect when you switch back to the matrix.

Two related pieces of information are *displayed* rather than edited here.
Per-type and per-context free-text notes, context type and area travel in the
project file — imported, migrated from v1, or written by another tool — and are
shown in the Inspector (§14); the v2 interface has no editor for them.
Per-type frequency totals are likewise not tabulated in this view; they are
available from the CSV/XLSX export and are what the Ford diagram normalises
against (§9).

---

## 11. Cell Annotations

Beyond the raw numeric matrix, individual cells (a specific type within a
specific context) often carry archaeologically important qualifications
that don't fit into a plain frequency count. The Cell Annotations feature
attaches structured metadata to a cell:

| Field | Purpose |
|---|---|
| **Certainty** | e.g. "certain" / "uncertain" — is the type identification itself in doubt? |
| **Fragmentation** | e.g. "complete" / "fragmented" — condition of the find |
| **Count range (min/max)** | when the exact count is uncertain, record a plausible range instead of a single number |
| **Inventory numbers** | link the cell to specific catalogued object IDs |
| **Notes** | free text for anything else worth recording |

### 11.1 Editing a single cell

A single cell is just a selection of size one: switch the matrix to
**Annotate** mode, drag over the one cell, and the annotation editor in the
Inspector fills in the fields above. Fields are merged with any existing
annotation on that cell — setting one field does not clear the others.

### 11.2 Batch-editing multiple cells

Switch the matrix to **Annotate** mode (§5.4) and drag a rectangular area across
multiple cells, or otherwise select a group of cells, then open the
annotation editor. If the selected cells already have **different**
existing values for a field (e.g. some "certain", some "uncertain"), that
field displays as blank/mixed in the editor.

**Only fields you actually type into or click are applied to the whole
selection when you save.** Fields you leave untouched are left exactly as
they were on each individual cell — a blank/mixed field does *not* mean
"clear this field everywhere." This lets you, for example, add a shared
note to a block of cells with otherwise heterogeneous certainty and
fragmentation values without disturbing that existing per-cell data.
Explicitly clearing a field (clicking into it and deleting its content) still
removes that field from the selection, as expected.

### 11.3 Where annotations live

Annotations are stored in the project keyed by canonical context/type
identity, not by display position — so reordering, filtering, or pinning
never disconnects an annotation from the cell it describes. They are
included in every project export (JSON, and represented in the Linked
Open Data exports as CIDOC-CRM–compatible statements, §15) but are
**not** currently tracked by Undo/Redo, which only covers structural
matrix operations (reordering, pinning, seriation runs) — keep that in
mind for particularly sensitive annotation edits.

---

## 12. Missing Data vs. Structural Absence

CombiTab makes an explicit, archaeologically important distinction:

- A matrix cell holding **`0`** means the type was **verifiably absent** —
  the context was excavated/examined and that type genuinely does not occur
  in it. This is a real, informative data point.
- A cell marked **"not recorded"** means the context simply **wasn't
  documented** with respect to that type — perhaps due to partial
  excavation, poor preservation, or incomplete publication. This is the
  *absence of information*, not information about absence, and treating it
  as a `0` would silently bias every downstream calculation.

### 12.1 Marking cells as "not recorded"

Select one or more cells (including via a rectangular area selection in
Annotate mode) and use the "not recorded" toggle. On import, common textual
tokens — `?`, `NA`, `N/A` (case-insensitive) — are automatically recognized
in CSV/TSV/XLSX source cells and imported as "not recorded" rather than as
zero, so you rarely need to mark these by hand after a well-formed import.

### 12.2 How missing cells are treated throughout the app

Every algorithm and metric in CombiTab that touches raw cell values has an
explicit, documented policy for missing cells (rather than an accidental
default):

- **Centroid seriation** (§6.1): excluded from both weights and weighted
  sums.
- **Correspondence analysis** (§8.2): imputed under the independence model
  via iterative proportional fitting — "no association" rather than a
  guessed pattern.
- **Concentration metric** (§7.1): excluded from both the weighted distance
  and the total weight.
- **Continuity metric** (§7.3): if inside a type's occurrence span, excluded
  from the denominator (treated as unknown, not as a gap).
- **Anti-Robinson index** (§7.2): pairwise-complete similarity — a missing
  cell drops out of that specific row-pair's comparison rather than
  contributing a spurious zero.

This consistency means you can mark uncertain data as missing with
confidence that it will be handled honestly everywhere in the app, rather
than needing to remember which views treat it as zero.

---

## 13. Stability Analysis — Mathematical Foundations

Any single seriation is a point estimate. The **Stability view** answers a
different, equally important question: *how much would this ordering
change if the underlying data had come out slightly differently?* — a
proxy for how confidently you can treat the resulting chronology as
well-supported by the evidence, versus a fragile artifact of a small
sample.

### 13.1 Bootstrap procedure

For a chosen number of replicates $B$ (default 200):

1. For each context $i$ with observed row total $n_i$ (its total artifact
   count) and observed type-proportion profile, draw a **multinomial
   resample** of size $n_i$ from that profile. This keeps the context's
   total assemblage size fixed while letting its *composition* vary
   according to sampling uncertainty — modeling the idea that if you'd
   excavated a slightly different (but equally sized) sample of that
   grave's contents, you might have recovered a somewhat different mix of
   types.
2. Recompute the first **correspondence analysis** dimension (§8) on this
   resampled matrix.
3. Align the sign of the resulting axis to the reference solution (CA axes
   are only defined up to sign; without this step, roughly half of all
   replicates would appear to run in the opposite chronological direction
   purely by an arbitrary sign flip).
4. Record each context's **rank** along this axis for this replicate.

Both the resampling and the sign-alignment reference use a **seeded
pseudo-random generator**, so a full stability run is exactly reproducible.

### 13.2 Interpreting the results

After all $B$ replicates, each context has a distribution of ranks across
replicates, summarized as:

- **Reference rank** — its rank in the actual (non-resampled) CA ordering.
- **Mean / median rank** across replicates.
- **5th–95th percentile interval** — a rank confidence band.
- **Standard deviation** of the rank distribution.

A **narrow** interval means the context's chronological position is robust
to sampling noise — different plausible re-excavations would place it in
much the same relative position. A **wide** interval flags a context whose
placement is more sensitive to exactly which finds happened to be present
— often contexts with very few finds, or with an unusual/ambiguous type
profile.

A single **global stability** figure summarizes the whole dataset as the
mean, across contexts, of $1 - \frac{\text{hi} - \text{lo}}{N_R - 1}$ (the
interval width normalized by the maximum possible rank spread), giving one
number to track as you refine your dataset or exclude problematic contexts.

Computation runs in a background Web Worker with progress reporting, so
larger replicate counts on large matrices don't freeze the interface, and a
run can be cancelled if you change your mind about the replicate count.

---

## 14. Project-Level Metadata

Beside the matrix itself, a CombiTab project carries descriptive information
that survives every export and re-import:

- **Project name** — also the base filename for exports.
- **Data type** — frequency vs. presence/absence (§3.3), which changes how
  several metrics are computed.
- **Context types** — a controlled vocabulary of context categories
  ("inhumation", "cremation", …), referenced by individual contexts.
- **Per-context fields** — context type, area, free-text notes.
- **Per-type fields** — material group, index-type flag, free-text notes.
- **History** — a provenance log. Every seriation run appends the method, its
  parameters (including the RNG seed, or the CA dimension used for ordering),
  an ISO timestamp and the resulting score. This is what makes a run
  reproducible after the fact, and it is the source the methods report draws on
  (§15.4).

Where these fields are visible: the Inspector shows the per-context and
per-type fields for whatever is selected, and the methods report and the Linked
Open Data exports draw on all of them.

**A limitation worth stating plainly:** the v2 interface offers editors for
material groups and type assignment (§10) and for cell annotations (§11), but
**not** for project name, data type, the context-type vocabulary, or the
free-text note fields. Those values arrive with the imported file — from a v1
project, a finds database export, or hand-editing the project JSON, which is a
documented, plainly structured format (§3.1) — and are then carried through
faithfully. If you need to change one today, edit the project JSON.

---

## 15. Export & Interoperability

The Export menu (with a live "busy" state and a brief confirmation
toast/error message on completion) offers:

### 15.1 Image exports
- **PNG** — raster snapshot of the current matrix view, rendered at 2×
  resolution for print quality.
- **SVG** — vector graphic, editable in Illustrator/Inkscape for further
  polishing before publication.
- **PDF** — vector PDF of the matrix, ready to include directly in a paper.

### 15.2 Data exports — exact structure

Both data exports serialize the matrix in the project's **current display
order** (`order.rows` / `order.cols`) — i.e. exactly the row/column sequence
you currently see on screen, not the underlying canonical import order.

- **CSV** — first column header `Context`, followed by one column per type
  in display order; one row per context in display order. Cells marked
  "not recorded" (§12) are written as **`?`**, distinguishing them from a
  genuine `0`. The file is written with **CRLF** line endings and a
  **UTF-8 byte-order mark (BOM)** prepended, specifically so that Excel on
  Windows displays umlauts and other non-ASCII characters correctly rather
  than mojibake — a common pain point with plain UTF-8 CSV on that
  platform. Values containing a comma, quote, or newline are quoted and
  escaped per standard CSV rules.

  ```csv
  Context,Fibula A,Bead,Spindle Whorl
  Grave_2,0,2,?
  Grave_1,3,0,1
  Grave_3,1,1,0
  ```

- **XLSX** — the identical table (same header row, same "not recorded" `?`
  convention) written to a single worksheet named **"Seriation"**, via
  SheetJS. The `xlsx` library is **loaded on demand** the first time you
  either import or export an XLSX file — it is not part of the app's
  initial JavaScript bundle, so users who never touch spreadsheets don't
  pay its (substantial) download cost.

Neither data export currently includes annotations, material-group
assignments, or missing-cell distinctions beyond the `?` marker — for a
fully lossless round trip including all of that, use the project JSON
export instead.

### 15.3 Project exports — exact structure

- **CombiTab JSON (v2)** — a direct `JSON.stringify` (pretty-printed, 2-space
  indent) of the complete internal `ProjectV2` object (§3.1): `contexts`,
  `types`, the dense `matrix`, `columnMetadata`/`rowMetadata` (keyed by
  name), `cellAnnotations` (keyed by canonical `"row:col"`), `missingCells`
  (same key scheme, present only if non-empty), `materialGroups`,
  `contextTypes`, the current `order`, `view`, `filters`, and the full
  `history` log. This is the **only** export format that losslessly
  round-trips absolutely everything, including annotations and filters —
  re-importing this file reproduces the project exactly as it was.
- **Legacy v1 JSON** — the same data translated back into the older
  `snake_case` v1 field schema (e.g. `rowRange` → `filter_row_range`), for
  interoperability with archives or tooling still on the original CombiTab
  format.

### 15.4 Documentation export

- **Methods (Markdown)** — a plain-text methods paragraph assembled from the
  project's `history` log (which seriation method(s) were run, with which
  random seed, in citation-ready English phrasing — e.g. "reciprocal
  averaging", "correspondence analysis seriation (CA dim. 1)") and the
  current bootstrap stability results, ready to paste into a paper's
  methods section.

### 15.5 Linked Open Data exports (CIDOC-CRM / CRMarchaeo)

- **Turtle (TTL)** and **JSON-LD** — both serializations are generated from
  one shared internal graph representation, so they are guaranteed to
  express exactly the same statements; choose whichever your downstream
  pipeline expects.

**Modeling (CRMarchaeo v2.x):**

| CombiTab concept | CIDOC-CRM / CRMarchaeo class or property |
|---|---|
| The dataset itself | `dcat:Dataset` |
| The site (project name) | `crm:E27_Site` — one shared site node for the whole project |
| A context (grave/feature) | `crmarchaeo:A2_Stratigraphic_Volume_Unit`, with `crm:P2_has_type` (context type), `crm:P53_has_former_or_current_location` (the site), and `ctb:seriationPosition` (its rank in the current seriation order) |
| An excavation unit | `crmarchaeo:A1_Excavation_Processing_Unit`, linked to its context via `crmarchaeo:AP5_removed_part_or_all_of` |
| An artifact type | `crm:E55_Type`, with its material group as `crm:P127_has_broader_term` and, for index types, a descriptive note |
| A material group | `crm:E57_Material` |
| An individual find | `crm:E22_Human-Made_Object` with `crm:P2_has_type` → its type, linked to its context via `crmarchaeo:AP21_contains` |

A cell with frequency $n$ produces **$n$ individual object identities**
(each its own `E22_Human-Made_Object`) rather than a single node with a
count attached — this lets each find be referenced individually elsewhere
in a Linked-Open-Data graph (e.g. by a later inventory-number linkage). For
very high frequencies this is capped (default 1000 objects per cell,
configurable via `maxObjectsPerCell`) with a documented fallback to an
aggregate representation using `crm:E54_Dimension` for the count instead —
the exported Turtle/JSON-LD explicitly notes in a header comment how many
cells were aggregated this way, rather than silently truncating data.

Cells marked "not recorded" (§12) and cells that are genuinely `0` both
produce **no** `E22` object for that context/type pair — a real absence is
not asserted as a find, and a missing value is not invented as one either.

#### 14.5.1 Stratigraphy: what is — and is not — exported

This is an important, deliberate limitation to understand before using the
LOD export in a digital-humanities pipeline that expects full
stratigraphic data:

**CombiTab does not capture observed stratigraphy.** It has no concept of a
Harris matrix, no notion of "context A cuts context B" or "context A is
stratigraphically above context B," and consequently the export emits
**no `crmarchaeo:AP13` relations** (the CIDOC-CRM/CRMarchaeo property used
to assert stratigraphic sequencing between volume units) — because
CombiTab was never given that information to begin with.

What *is* exported is only `ctb:seriationPosition`: the context's rank in
the **current seriation display order** — a relative position derived
purely from artifact co-occurrence patterns (§6), not from any excavated
stratigraphic observation. This is exported deliberately as an **inferred,
directionless sequence**:

- It is *inferred* — a mathematical ordering by assemblage similarity, not
  an observed physical relationship. Two contexts adjacent in the
  seriation order are similar in content; that is evidence *suggestive of*
  chronological proximity, not proof of it, and it is not the same kind of
  statement as an excavator's stratigraphic observation.
- It carries **no asserted temporal direction** — seriation produces a
  sequence (a linear ordering), not a "before/after" arrow. Nothing in the
  export claims context #3 is *earlier* than context #7; it only says
  they are three and seven positions apart in the derived ordering.

**Practical consequence:** if your project has independently observed
stratigraphic relationships (cuts, layers, above/below relations from
excavation), CombiTab does not store or export them, and you will need to
maintain and reconcile that information in a separate stratigraphy tool
(e.g. a Harris matrix application) and combine it with CombiTab's
`ctb:seriationPosition` values yourself downstream — typically using the
seriation order as a *cross-check or refinement* of the stratigraphic
phasing, not as a replacement for it. The CRMarchaeo `AP13` property is
explicitly the documented extension point for this in the data model — the
export code notes it as "the point to extend to once stratigraphic
observations are available" — but it is intentionally **not implemented**
in the current version, since CombiTab has no input mechanism for
recording that data in the first place. In short: **CombiTab is a
co-occurrence–based seriation tool, not a stratigraphy/Harris-matrix
tool**, and its Linked Open Data export is honest about that boundary
rather than overstating what the underlying analysis supports.

---

## 16. Sharing, Autosave & Offline Use

### 16.1 Share link
The **Share** button copies a URL encoding the complete current state — project,
display order, active filters, annotations, missing-value marks and the active
tab. Two details matter:

- The state goes into the **URL fragment** (after the `#`), which browsers never
  transmit to a server. Sharing a link therefore does not upload your data
  anywhere; it travels only through whatever channel you paste it into.
- It is compressed with the browser's native `CompressionStream` (gzip) and
  encoded base64url. Above **16 000 characters** CombiTab refuses to pretend and
  tells you the project is too large for a link, asking you to share the project
  file instead — some browsers and mail gateways silently truncate longer URLs.

This is the convenient route for "look at this filtered subset and tell me what
you think"; the project file is the route for anything larger or archival.

### 16.2 Autosave
When the page is hidden or closed, the current project is written to the
browser's **IndexedDB**; on the next start a banner offers to restore it, naming
the project and the time it was saved. Every access is fault-tolerant: if
IndexedDB is unavailable — private browsing, a locked-down environment — autosave
disables itself silently rather than interrupting you.

Autosave is a safety net against an accidental close or a crash, **not a
backup**. It holds exactly one state, in one browser profile, on one machine.
Export a project file for anything you would be unhappy to lose.

### 16.3 Offline behaviour
The web version, once installed as a PWA (§2.5), caches its app shell and
bundled assets in a service worker and is fully usable without a connection. The
desktop edition needs no such mechanism: everything, fonts included, is shipped
inside the package, and its content-security policy forbids outbound connections
outright.

In neither case does project data leave the machine. It lives in the files you
explicitly load and export, and in the autosave record — CombiTab has no backend
to send it to.

---

## 17. Performance & Large Datasets

CombiTab is designed to comfortably handle matrices with **100+ contexts
and 100+ types**:

- The renderer's reordering is O(1) per drag (§5.3), not proportional to
  cell count.
- Seriation and stability computations run in background **Web Workers**,
  keeping the UI responsive even during expensive runs, with the ability to
  cancel an in-flight computation.
- The quality-score anti-Robinson index — the most expensive metric,
  $O(N_R^2 \times N_C)$ — is likewise computed off the main thread for
  large matrices.

**Practical tips:**

- Use **filters** (§5.6) to work on a manageable subset while fine-tuning a
  particular region, then clear the filter once satisfied.
- Run **centroid** or **CA** first on the full (or filtered) dataset; reserve
  **iterative optimization** for a final polishing pass, since it starts
  from — and can only improve on — the centroid result.
- If the interface ever feels sluggish on a large matrix, open the
  **performance HUD** (`Shift+P`) to see live FPS, draw time, and cell
  counts — useful both for diagnosing your own hardware/browser
  combination and for reporting a reproducible performance issue.

---

## 18. Accessibility

- **Light and dark themes**, both designed to meet contrast requirements, with
  the choice remembered between sessions.
- **Colourblind-safe palette** for material groups (Okabe-Ito), applicable to a
  whole project with one button, plus simulation of deuteranopia, protanopia
  and tritanopia and an explicit warning naming colour pairs that fall below
  the CIELAB ΔE discrimination threshold (§10). The underlying colour core
  performs the dichromacy simulation in linear light, following Viénot et al.
  (1999), rather than the naive sRGB approximation that flatters the result.
- **Keyboard and screen reader**: a skip link to the main content, a
  `:focus-visible` focus ring throughout, landmark regions, tabs implemented to
  the WAI-ARIA tabs pattern (roving `tabindex`, arrow keys, Home/End), errors
  announced via `role="alert"`, `<html lang>` following the interface language,
  and the matrix canvas exposed as `role="img"` with a label naming its
  dimensions and pointing to the CSV/Excel export for the cell values — a
  canvas cannot be read by a screen reader, so the label says where the data
  can actually be obtained.
- **`prefers-reduced-motion`** disables non-essential transitions.
- **Canvas-2D fallback** — without WebGL2 the matrix renders through a 2D
  context using an exactly reproduced colour model rather than failing to
  display, at reduced performance on very large matrices.
- **Cross-browser export** is covered by its own tests (UTF-8 BOM on CSV for
  Excel on Windows, XML prolog and namespace on SVG, `/MediaBox` on PDF,
  transliterated portable filenames, a `window.open` fallback where
  `<a download>` is unavailable). Note that Firefox remains the most reliable
  browser for triggering downloads; some Chromium configurations interfere
  with download prompts.

---

## 19. Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl/Cmd + Z` | Undo (structural matrix operations) |
| `Ctrl/Cmd + Shift + Z` or `Ctrl/Cmd + Y` | Redo |
| `Shift + P` | Toggle the performance HUD |
| `←` `→` `Home` `End` | Move between tabs while the tab bar has focus |

All shortcuts are automatically suppressed while a text field, text area, or
other editable control has focus, so they never interfere with typing notes,
inventory numbers, or search terms.

---

## 20. End-to-End Workflows

### 20.1 First analysis of a new cemetery dataset

1. **Import** a CSV/TSV/XLSX export from your finds database (or an
   existing CombiTab project).
2. In the **Metadata** view (§10), assign each type to a material group and
   mark any known index types.
3. Confirm the **data type** (frequency vs. presence/absence, §3.3) is what
   your import implies — it is set at import time and changes how several
   metrics behave.
4. Run **centroid seriation** as a first pass and note the quality score.
5. Optionally cross-check with **CA seriation**; if the two largely agree,
   that's a good sign the ordering reflects real structure rather than an
   artifact of one particular method.
6. Review contexts and types that look like outliers in the **CA scatter
   plot** (§8.3) and in the **Ford diagram** (§9), and check whether they are
   archaeologically unusual or a data problem.
7. **Pin** any independently well-dated contexts (coins, dendrochronology,
   radiocarbon) as fixed anchors, then re-run seriation so the algorithm
   respects them.
8. Run **iterative optimization** as a polishing pass.
9. Check the **Stability view** to see which contexts have well- vs.
   poorly-determined positions; consider whether poorly-determined contexts
   need more data or should be flagged as tentative in your publication.
10. **Annotate** any cells with uncertain identifications, fragmentary
    finds, or inventory references worth recording.
11. **Export** a methods paragraph, a publication-quality PNG/SVG/PDF of the
    final matrix, and a CombiTab project JSON as your working archive.

### 20.2 Collaborative review of a specific sub-question

1. Apply a **filter** to isolate the material group or row/column range in
   question.
2. Use **Focus mode** on a specific context or type of interest.
3. Generate a **Share link** — it captures the active filter and focus, so
   your collaborator opens the link and sees exactly the same narrowed
   view.
4. Discuss and annotate cells directly; annotations are visible to anyone
   who subsequently opens the same project file.

### 20.3 Re-analysis / methodological reproducibility

1. Open a previously exported CombiTab project JSON.
2. Read the project's **history** log (§14) — it records the exact method,
   its parameters and the seed of every seriation run that was performed.
3. Re-run the same method with the same seed to obtain a **byte-identical**
   result — all randomized steps in CombiTab (centroid's start vector,
   iterative optimization's starting point, bootstrap resampling) use a
   seeded generator specifically so this is possible.
4. Export the **Methods** report to get a ready-to-cite paragraph
   describing exactly what was done.

---

## 21. Architecture Notes

For readers with a technical interest in how CombiTab is built:

- **Frontend:** React + TypeScript, bundled with Vite.
- **Rendering:** a hand-written WebGL2 renderer for the matrix grid (with a
  Canvas-2D fallback), chosen over an off-the-shelf charting library because
  it needs to stay fluid while dragging and zooming matrices with tens of
  thousands of cells.
- **Heavy computation** (seriation runs, quality scoring on large matrices,
  bootstrap stability) executes in dedicated **Web Workers**, so the main
  thread — and therefore the UI — never blocks on a long computation, and
  in-flight computations can be cleanly cancelled.
- **No backend.** CombiTab is a fully static single-page application; all
  computation happens client-side. Project persistence is via explicit file
  export/import, the autosave layer (IndexedDB), and share links (state
  encoded in the URL fragment).
- **Minimal dependencies.** React is the only unconditional runtime
  dependency; SheetJS is loaded lazily and only when XLSX is actually used.
  SVD, correspondence analysis, PDF generation, i18n, colour metrics and icon
  generation are written in-house — which is why the security surface and the
  bundle both stay small.
- **Desktop shell.** Electron, loading the app through a custom `app://`
  scheme rather than `file://` so that ES module workers, IndexedDB and
  `CompressionStream` all behave as they do on the web (§2.2), with context
  isolation, sandboxing and a restrictive content-security policy. A
  ten-point smoke test (`npm run smoke`) starts a headless window and checks
  exactly the properties that break when an app moves into a shell; it runs in
  CI and against the finished packages.
- **Internationalization:** a small dictionary-based i18n system (German and
  English), with an automated test ensuring no UI string is missing or
  empty in either language.
- **Testing:** an extensive suite of unit and integration tests, including
  numerical validation of the correspondence analysis implementation
  against Greenacre's published reference values.
- **Offline support:** a service worker implements an app-shell caching
  strategy (network-first for navigation, stale-while-revalidate for
  hashed static assets), enabling full offline use once installed as a PWA.

---

## 22. Troubleshooting

**Downloads/exports don't trigger in my browser.**
Some Chromium-based browsers (or certain extensions) can silently interfere
with file-download prompts. Firefox has proven the most consistently
reliable browser for CombiTab's export downloads; if an export seems to do
nothing in Chrome/Edge, try Firefox or check your browser's download
permission settings for the site.

**The matrix looks blocky, slow, or doesn't render at all.**
Confirm your browser/GPU combination supports WebGL2 — CombiTab
automatically falls back to a Canvas-2D renderer if not, which works but is
slower on large matrices. Open the performance HUD (`Shift+P`) to check the
active rendering backend and current frame times.

**I can't reorder a row/column by dragging it.**
Check whether it's **pinned/fixed** — pinned elements cannot be dragged by
design (§5.3), to protect anchor contexts from accidental reordering.
Unpin it via the Inspector panel if you need to move it.

**A batch annotation edit didn't change the field I expected.**
Only fields you actually click into or type into are applied to a
multi-cell selection (§11.2) — this is intentional, to avoid overwriting
existing per-cell data that happens to differ across the selection. If you
want to clear a field across the whole selection, explicitly click into it
and delete its contents before saving.

**A cell reads as empty (0) but I know the data wasn't actually collected
for it.**
Mark it (or a whole selection) as **"not recorded"** (§12) rather than
leaving it as a numeric zero — the two are treated very differently
throughout the app's algorithms and metrics.

**Windows says "Windows protected your PC", or macOS says CombiTab "is
damaged and can't be opened".**
Neither is a malware finding. The packages are not code-signed (§2.2). On
Windows, click the small "More info" link — only then does "Run anyway"
appear. On macOS, right-click the app and choose "Open" instead of
double-clicking; if the "is damaged" message appears anyway, macOS's quarantine
flag is the cause and `xattr -cr /Applications/CombiTab.app` clears it. Full
walkthrough in [INSTALLATION.md](INSTALLATION.md).

**The Linux AppImage fails with `dlopen(): error loading libfuse.so.2`, or its
window never appears.**
Both are Ubuntu 24.04 changes affecting every AppImage, not just CombiTab:
FUSE 2 is gone (`sudo apt install libfuse2t64`, or run the AppImage with
`--appimage-extract-and-run`) and unprivileged user namespaces are restricted
(`--no-sandbox` as a workaround). The `.deb` package avoids both — it ships the
AppArmor profile and installs it for you.

**Correspondence analysis, the bootstrap or the score silently produce nothing
in a locally opened copy.**
You are almost certainly opening `dist/index.html` over `file://`. Chromium
refuses to load ES module workers from `file://`, and all three of those
computations run in workers. Serve `dist/` over HTTP (`npm run preview`, or any
static server) or use the desktop edition, which exists partly to solve exactly
this (§2.2).

**I restored an autosaved session and my newest changes are missing.**
Autosave writes when the page is hidden or closed and keeps exactly one state
per browser profile (§16.2). A hard crash before that write, or work done in a
different browser or on a different machine, is not covered. Export a project
file for anything you would be unhappy to lose.

**My share link is rejected as too large.**
Above 16 000 characters CombiTab declines to produce a link rather than hand
you one that some mail gateway will truncate (§16.1). Send the project file
instead — or apply a filter first, since the link encodes the filtered state.

**My keyboard shortcut isn't triggering / typed the wrong character.**
Shortcuts are suppressed automatically while a text field has focus; if a
shortcut also isn't working outside a text field, check whether your
browser or OS has bound the same combination to something else.

---

---

*This guide describes CombiTab 2.0.0. Corrections and questions are welcome
at [github.com/oeai-dac/CombiTab/issues](https://github.com/oeai-dac/CombiTab/issues).*
