# CombiTab 2 — Quick Start Guide

*Archaeological seriation and combination-table analysis*

**Version:** 2.0.0 · **License:** MIT · **Author:** Christian Gugl, Austrian
Archaeological Institute, Austrian Academy of Sciences (ÖAW)

**Written with:** Anthropic's Claude, in Claude Code. Large parts of the
application and of this documentation were developed with its assistance. The
direction of the work, the archaeological and methodological decisions, the
testing and the responsibility for the result lie with the author.

> Looking for more depth? [**GUIDE.md**](GUIDE.md) is the complete reference,
> including the mathematics behind the seriation methods, the quality metrics
> and the bootstrap. Step-by-step installation instructions for people without
> a technical background are in [**INSTALLATION.md**](INSTALLATION.md) (German).

---

## 1. What is CombiTab?

CombiTab is a tool for **archaeological seriation**: ordering contexts
(typically graves) and artefact types by their co-occurrence patterns to reveal
chronological sequences and cultural groupings. You feed it a matrix of
contexts × types — frequencies or presence/absence — and it helps you reorder
that matrix, by hand or automatically, until similar assemblages sit close
together and the classic "battleship curve" pattern emerges.

Two things are worth knowing up front:

- **Everything stays on your machine.** There is no server, no account and no
  telemetry. The desktop edition is additionally locked down by a
  content-security policy that forbids any outbound connection at all.
- **It comes in two flavours from the same code base:** a desktop application
  for Windows, macOS and Linux, and a web version that runs in the browser and
  keeps working offline once loaded.

The interface is available in German and English (the **DE/EN** button in the
header); the initial choice follows your browser or system language.

---

## 2. Getting CombiTab

There are three routes. Pick the first one that fits.

### a) Just try it — no installation

Open **<https://oeai-dac.github.io/CombiTab/>**.

All computation happens in your own browser; nothing is uploaded. This is the
fastest route, the right one on a managed work computer where you may not
install software, and the recommended route on Intel Macs, for which no package
is shipped.

### b) Install the desktop application (recommended for regular work)

Download the file for your system from the
[**Releases**](https://github.com/oeai-dac/CombiTab/releases) page, under
"Assets" of the newest version:

| Your system | File |
|---|---|
| Windows 10 / 11 | `CombiTab-2.0.0-Setup-x64.exe` |
| Windows, without installing | `CombiTab-2.0.0-portable-x64.exe` |
| macOS with Apple silicon (M1–M4) | `CombiTab-2.0.0-arm64.dmg` |
| Linux, any distribution | `CombiTab-2.0.0-x86_64.AppImage` |
| Ubuntu 22.04+, Debian 12+, Mint | `combitab_2.0.0_amd64.deb` |
| Fedora, openSUSE, RHEL | `combitab-2.0.0.x86_64.rpm` |

**Expect a security warning on the first launch.** The packages are not
digitally signed — a certificate costs several hundred euros a year and would
not improve the software itself, only register the publisher with Microsoft and
Apple. On Windows, click "More info" → "Run anyway"; on macOS, right-click the
app and choose "Open" instead of double-clicking it.
[**INSTALLATION.md**](INSTALLATION.md) walks through each system in detail,
including the Ubuntu 24.04 quirks around FUSE and the sandbox, and the
`xattr -cr` fix for macOS's misleading "is damaged" message.

The Windows installer needs **no administrator rights** — it installs into your
user profile, which works on managed institutional machines. The portable `.exe`
installs nothing at all and runs from a USB stick.

### c) Build from source (developers)

```bash
npm ci             # install dependencies (Node.js 18+, developed on Node 22)
npm run dev        # Vite dev server, typically http://localhost:5173
npm run build      # type-check + production build into dist/
npm test           # the full test suite
npm run electron   # build and launch inside the desktop shell
npm run dist       # build installer packages for the current system
```

`dist/` is a plain static site and can be served by any web server. Packaging,
release and code-signing details are in [BUILD.md](BUILD.md). Note that Windows
and macOS packages cannot be built on Linux; they are produced by the release
workflow on real runners.

### Installing the web version as an offline app (PWA)

Opened in Chrome or Edge, CombiTab offers an **install** button (also in the
address bar). Once installed it runs in its own window and works fully offline.
Touch input is supported throughout the matrix — panning, drag-reordering and
rectangular selection all work with a finger or stylus — so a tablet is a
practical field or storeroom device. Zoom on touch is via the **cell size**
slider rather than pinch gestures.

---

## 3. Getting your data in

On startup CombiTab loads a **demo dataset**, so every feature can be tried
immediately without preparing anything.

To load your own data, use **"Load file…"** in the header or simply **drag and
drop** a file anywhere into the window:

| Format | Notes |
|---|---|
| `.csv` / `.tsv` | Raw contexts × types table (frequencies or 0/1 presence) |
| `.xlsx` | The same, read from the first worksheet |
| `.combitab.json` | A CombiTab v2 project — matrix, metadata, annotations, view state |
| `.json` (v1) | A project from CombiTab 1 — imported and migrated automatically |

Cells reading `?`, `NA` or `N/A` are imported as **"not recorded"**, which
CombiTab keeps strictly separate from a true zero (see §6).

---

## 4. The five views

The tab bar at the top switches between five views. They share one selection: a
context you hover or click in the matrix lights up in the CA plot, the Ford
diagram and the stability chart as well.

### Matrix
The main workspace, and where seriation actually happens. A WebGL2-rendered
grid in which **cell colour** is the material group of the column and
**saturation** is the cell's value.

- **Zoom** with the mouse wheel (around the cursor) or the **cell size** slider;
  **pan** by dragging the canvas. **Fit** resets the view.
- **Hover** shows a live inspector; **click** selects a cell, context or type
  and cross-highlights its row and column.
- **Drag-reorder:** grab a row or column *label* and drag it. This is instant
  even on large matrices.
- **Pin (fix) elements** from the Inspector: pinned contexts and types keep
  their position, and seriation orders only the free elements around them —
  which is how you anchor independently dated graves.
- **Undo/redo** covers every structural change.
- The **minimap** at the edge gives you a viewport rectangle for navigating
  large matrices.
- **Mode: Navigate / Annotate.** Switching to *Annotate* turns dragging into
  rectangular cell selection for batch annotation (§6).
- **Filters** in the sidebar: material-group chips, row/type ranges, "hide empty
  rows/columns", and **Focus on selection**, which narrows the view to the
  neighbourhood of the current selection. A banner reminds you that reordering,
  seriation, pinning and annotation act on the *visible* subset and are written
  back into the full project.

**Seriation lives in the matrix toolbar**, not in a tab of its own. Choose a
**method** — *centroid* (reciprocal averaging), *correspondence analysis* (with
a selectable CA dimension) or *iterative optimization* — and press **Seriate**.
Runs happen in a background worker, so the interface stays responsive and a run
can be cancelled. A live **quality score** reports concentration, continuity,
the anti-Robinson index and a combined total, so you can watch the ordering
improve.

### Correspondence analysis
A scatter plot (biplot) of contexts and types on freely selectable CA axes,
plus a **scree plot** of explained inertia. Useful for spotting outliers and for
checking the seriation visually — a pronounced arch on dimension 2 (the Guttman
effect) is the signature of a clean seriation structure. The implementation is
an exact SVD, cross-checked against Greenacre's published reference values.

### Ford diagram
The classical **battleship curves**: for every context, bar width is the share
of that type in the context's inventory, with contexts in the current seriation
order. This is the figure most seriation publications actually print.

### Stability
Bootstrap robustness. The type frequencies of each context are resampled
repeatedly (100, 200 or 500 replicates) and CA dimension 1 is recomputed each
time. The **caterpillar plot** shows, per context, the 90 % rank interval, the
median rank and the reference rank of your current ordering; a global stability
figure summarises it. Narrow intervals mean a chronologically well-supported
position; wide intervals call for caution when dating that context. The result
also feeds into the methods export.

### Metadata
Curation of the classification, in two parts:

- **Material groups** — create, rename, recolour and delete them. Group colour
  drives the cell colouring, the matrix legend and the filters. One button
  applies a **colourblind-safe palette** (Okabe-Ito); a preview shows every
  colour under deuteranopia, protanopia and tritanopia, and CombiTab warns about
  pairs that are hard to tell apart.
- **Type table** — searchable; assign each type to a material group and mark
  **lead types** (chronologically diagnostic types).

Project-level information — project name, data type, context types, per-row and
per-column notes — travels in the project file and is shown in the Inspector.

---

## 5. A typical session

1. **Import** your contexts × types table, or open an existing project.
2. In **Metadata**, assign types to material groups and mark lead types.
3. Back in **Matrix**, run **seriation** — start with the centroid method or CA
   — and watch the quality score.
4. **Cross-check** in the CA view and the Ford diagram: if two independent
   methods largely agree, the ordering is more likely to reflect real structure
   than one method's artefact.
5. **Fine-tune by hand:** drag rows and columns, pin well-dated anchor contexts,
   use focus mode to inspect a neighbourhood.
6. **Annotate** ambiguous or noteworthy cells.
7. **Check robustness** in the Stability view.
8. **Export** the matrix figure, the sorted data, the project file and the
   methods report.

Every structural change — reordering, seriation runs, pinning — is covered by
undo/redo.

---

## 6. Annotating cells, and "not recorded"

Switch the matrix to **Annotate** mode and drag a rectangle over the cells you
want to describe. The annotation editor sets **certainty** (certain / uncertain
/ questionable), **fragmentation**, a **count range**, **inventory numbers** and
free-text **notes** for the whole selection at once. Annotated cells carry a
small marker.

When you edit a batch of cells whose existing values differ, **only the fields
you actually filled in are written**; fields you leave empty keep their
per-cell values instead of being flattened. To clear a field across the whole
selection, clear it explicitly.

The same panel marks cells as **"not recorded"**. This is deliberately distinct
from a value of 0: a zero asserts that the type does not occur in that context,
whereas "not recorded" says nothing was documented. The matrix draws such cells
with a diagonal cross, the Inspector reports present / absent / not recorded
counts, and every algorithm and metric treats the two cases differently. Export
writes them back as `?`.

---

## 7. Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl/⌘ + Z` | Undo |
| `Ctrl/⌘ + Shift + Z` or `Ctrl/⌘ + Y` | Redo |
| `Shift + P` | Toggle the performance HUD (frame and draw time) |
| `←` `→` `Home` `End` | Move between tabs while the tab bar has focus |

Shortcuts are suppressed while you are typing in a text field, so they never
interfere with notes, inventory numbers or the type search.

---

## 8. Sharing, autosave and offline use

**Share link.** The **Share** button copies a URL encoding the complete state —
project, ordering, filters, annotations, missing-value marks and the active tab
— into the URL *fragment*, which browsers never send to a server. A colleague
opening the link sees exactly your view, with no file transfer. Beyond roughly
16 000 characters CombiTab tells you the project is too large for a link and
asks you to share the project file instead.

**Autosave.** When you hide or close the page, the current project is written to
the browser's IndexedDB; on the next start a banner offers to restore it. This
is a safety net against an accidental close, not a backup — export a project
file for that. In private-browsing mode autosave silently disables itself.

**Offline.** The installed web version caches its app shell in a service worker
and remains fully usable without a connection. The desktop edition has no
network dependency at all.

---

## 9. Export

The **Export** menu in the header is reachable from every tab:

| Group | Formats | Use |
|---|---|---|
| Image (matrix) | PNG (×2), SVG, PDF | Publication figures; SVG and PDF are true vector output |
| Data (sorted) | CSV, XLSX | Re-analysis in R, Excel or elsewhere, in seriation order |
| Project | CombiTab v2 `.combitab.json`, v1-compatible `.json` | Archiving, collaboration, round-trip |
| Science | Methods report (Markdown, DE/EN) | A ready-to-cite paragraph with dataset figures, method, quality metrics, CA inertia, bootstrap stability and the citation |
| Linked Open Data | Turtle, JSON-LD | CIDOC-CRM / CRMarchaeo for semantic-web and digital-humanities pipelines |

Image exports use the current display order, label pinned elements in red and
carry the material-group legend. A short confirmation appears after each export.
In the desktop edition, exports open a native save dialog instead of dropping
the file into the downloads folder.

If a download silently fails to appear in a Chromium-based browser, try Firefox
— it has proven the most reliable for download prompts.

---

## 10. Tips for large datasets (100+ rows and columns)

- Use **filters** to work on a manageable subset, then clear them once that
  section is fine-tuned.
- On a first pass use **centroid** or **CA**; keep **iterative optimization**
  for polishing, since it refines an existing order rather than finding one.
- If the interface ever feels sluggish, `Shift+P` opens the performance HUD with
  live frame and draw times and the active rendering backend — useful for
  diagnosing your hardware and for filing a reproducible performance report.
- Without WebGL2, CombiTab falls back to a Canvas-2D renderer with an identical
  colour model. Everything keeps working; only very large matrices get slower.

---

## 11. Getting help

CombiTab is open source under the MIT licence. Please report problems at
[github.com/oeai-dac/CombiTab/issues](https://github.com/oeai-dac/CombiTab/issues),
ideally with your operating system and version, the file you downloaded and the
exact wording of the error message. Questions about the archaeological
methodology are best directed to the author, Christian Gugl (Austrian
Archaeological Institute, ÖAW).

---

*This guide describes CombiTab 2.0.0. The full reference is
[GUIDE.md](GUIDE.md); the project overview is in the
[README](../README.md).*
