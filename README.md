# CombiTab

**Archaeological Seriation & Combination Table Tool**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)

A Streamlit-based application for creating, editing, and analyzing archaeological seriation diagrams (combination tables / Kombinationstabellen).

## Overview

CombiTab enables archaeologists to establish relative time sequences based on co-occurrence patterns of artifact types across different archaeological contexts. The application combines powerful seriation algorithms with interactive visualization tools for professional chronological analysis.

The method rests on two fundamental assumptions: artifact types have limited lifespans (introduction → peak popularity → decline), and contexts sharing similar assemblages are likely contemporary. Proper ordering creates the characteristic diagonal pattern of well-seriated combination tables.

## Key Features

### 📊 Three Visualization Styles
- **Classic Squares**: Traditional black/white representation for publications
- **Battleship Bars**: Horizontal bars scaled by frequency
- **Sized Dots**: Circular markers for clear quantity distinction

### 🔢 Seriation Algorithms

**Centroid Method**
```
Centroid = Σ(position × value) / Σ(value)
```
Intuitive sorting by weighted average position of occurrences.

**Correspondence Analysis (CA)**
- Multivariate statistical technique revealing underlying data structure
- First dimension typically corresponds to chronological gradient
- Biplot visualization of context-type relationships
- Scree plot for dimension analysis

**Iterative Seriation**
- Repeated application until convergence
- Supports both centroid and CA methods
- Respects fixed elements during sorting

### 📈 Quality Metrics

| Metric | Description |
|--------|-------------|
| Concentration Index | Measures diagonal clustering (0–1) |
| Anti-Robinson Index | Detects gaps in type distributions |
| Type Continuity | Density of types within their span |
| Overall Quality Score | Weighted combination (scores >0.8 = good) |

### 🏷️ Rich Metadata System

**Column Metadata (Artifact Types)**
- Material group assignment (Ceramic, Metal, Glass, Bone, Stone, Organic)
- Index type (Leittyp) flagging
- Position locking for sorting constraints

**Row Metadata (Contexts)**
- Context type classification (Grave, Pit, Ditch, Layer, Posthole, Structure)
- Excavation area recording
- Position locking

**Cell Annotations**
- Certainty levels (Certain, Uncertain, Questionable)
- Fragmentation status (Complete, Fragmentary, Unknown)
- Inventory numbers and notes

### 🎨 Interactive Features
- Zoom, pan, and reset navigation
- Hover tooltips with full cell information
- Color-coded material groups in headers
- Manual drag-and-drop ordering
- Undo/Redo system (20 steps)

### 🔍 Filtering & Focus Mode
- Material group filtering
- Range-based row/column focus
- Hide empty rows/columns option
- Edit mode for large dataset performance

### 💾 Export Options

**Image Formats**
- PNG (adjustable DPI)
- SVG (vector)
- PDF (vector)

**Data Formats**
- CSV (sorted matrix)
- Excel with metadata (multi-sheet)
- Excel with annotations
- JSON project files (complete state)

## Correspondence Analysis

The CA module provides:
- **Biplot**: Joint display of contexts (points) and types (arrows)
- **Dimension Interpretation**: Chronological and functional axes
- **Scree Plot**: Eigenvalue distribution for dimension selection
- **Material Group Coloring**: Pattern recognition across categories

## Platform Support

| OS | Status |
|----|--------|
| Windows 10/11 | ✅ Supported |
| macOS 12+ | ✅ Supported |
| Linux (Ubuntu 20.04+) | ✅ Supported |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl + Z | Undo |
| Ctrl + Y | Redo |
| Ctrl + S | Export project |

## Documentation

Full documentation including mathematical foundations and methodology is available in the `docs/` folder.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed at the Austrian Academy of Sciences for archaeological seriation research.

---

**CombiTab** — Professional seriation analysis for archaeological chronology
