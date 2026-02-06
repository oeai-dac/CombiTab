"""
CombiTab - Archaeological Seriation & Combination Table Tool
Version: 1.0

Designed by Christian Gugl (christian.gugl@oeaw.ac.at)
Austrian Academy of Sciences (ÖAW)
Coded with the assistance of Anthropic Claude

License: MIT License
Copyright (c) 2026 Christian Gugl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import io
import json
import base64
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List

# Page configuration
st.set_page_config(
    page_title="CombiTab",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8B4513;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stDataFrame {
        font-size: 0.8rem;
    }
    .cell-editor {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .annotation-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Visualization constants
CERTAINTY_STYLES = {
    'certain': {'alpha': 1.0, 'hatch': None, 'label': '● Certain'},
    'uncertain': {'alpha': 0.7, 'hatch': '///', 'label': '◐ Uncertain'},
    'questionable': {'alpha': 0.4, 'hatch': 'xxx', 'label': '○ Questionable'}
}

FRAGMENTATION_MARKERS = {
    'complete': {'marker': 's', 'label': '■ Complete'},
    'fragmentary': {'marker': '^', 'label': '▲ Fragmentary'},
    'unknown': {'marker': 'o', 'label': '● Unknown'}
}


# ============================================================
# DOWNLOAD HELPER FUNCTIONS (Base64 workaround for reliable downloads)
# ============================================================

def create_download_link(data, filename, mime_type, link_text="⬇️ Download"):
    """
    Create a reliable HTML download link using base64 encoding.
    This workaround avoids the st.download_button rerun issues.
    
    Args:
        data: bytes or str data to download
        filename: name for the downloaded file
        mime_type: MIME type of the file
        link_text: text to display on the link
    
    Returns:
        HTML string with download link
    """
    if isinstance(data, str):
        b64 = base64.b64encode(data.encode()).decode()
    else:
        b64 = base64.b64encode(data).decode()
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" style="display: inline-block; padding: 0.5rem 1rem; background-color: #FF4B4B; color: white; text-decoration: none; border-radius: 0.3rem; font-weight: 500; text-align: center;">{link_text}</a>'
    return href


def render_download_link(data, filename, mime_type, link_text="⬇️ Download"):
    """Render a download link using st.markdown"""
    html = create_download_link(data, filename, mime_type, link_text)
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# DATA CLASSES FOR PROJECT STRUCTURE
# ============================================================

@dataclass
class ColumnMetadata:
    """Metadata for artifact type columns"""
    name: str
    material_group: str = "Unassigned"
    color: str = "#808080"
    is_index_type: bool = False  # Leittyp
    is_fixed: bool = False
    is_visible: bool = True
    notes: str = ""


@dataclass 
class RowMetadata:
    """Metadata for context rows"""
    name: str
    context_type: str = "Unassigned"
    area: str = ""
    is_fixed: bool = False
    is_visible: bool = True
    notes: str = ""


@dataclass
class CellAnnotation:
    """Annotation for individual cells"""
    certainty: str = "certain"  # certain, uncertain, questionable
    fragmentation: str = "unknown"  # complete, fragmentary, unknown
    count_min: Optional[int] = None  # minimum count (for ranges)
    count_max: Optional[int] = None  # maximum count (for ranges)
    inventory_numbers: str = ""  # comma-separated inventory numbers
    notes: str = ""


@dataclass
class Project:
    """Complete project data structure"""
    name: str = "Untitled Project"
    matrix: Optional[pd.DataFrame] = None
    data_type: str = "presence_absence"  # presence_absence or frequency
    column_metadata: dict = field(default_factory=dict)
    row_metadata: dict = field(default_factory=dict)
    cell_annotations: dict = field(default_factory=dict)
    material_groups: dict = field(default_factory=lambda: {
        "Unassigned": "#808080",
        "Ceramic": "#CD853F",
        "Metal": "#4682B4",
        "Glass": "#20B2AA",
        "Bone/Antler": "#DEB887",
        "Stone": "#696969",
        "Organic": "#8B4513"
    })
    context_types: list = field(default_factory=lambda: [
        "Unassigned", "Grave", "Pit", "Ditch", "Layer", "Posthole", "Structure"
    ])


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize session state variables"""
    if 'project' not in st.session_state:
        st.session_state.project = Project()
    if 'row_order' not in st.session_state:
        st.session_state.row_order = []
    if 'col_order' not in st.session_state:
        st.session_state.col_order = []
    if 'selected_row' not in st.session_state:
        st.session_state.selected_row = None
    if 'selected_col' not in st.session_state:
        st.session_state.selected_col = None
    if 'ca_info' not in st.session_state:
        st.session_state.ca_info = None
    if 'quality_history' not in st.session_state:
        st.session_state.quality_history = []
    # Visualization options
    if 'viz_style' not in st.session_state:
        st.session_state.viz_style = "classic"
    if 'cell_size' not in st.session_state:
        st.session_state.cell_size = 0.4
    if 'show_values' not in st.session_state:
        st.session_state.show_values = False
    if 'show_colors' not in st.session_state:
        st.session_state.show_colors = True
    if 'show_certainty' not in st.session_state:
        st.session_state.show_certainty = True
    if 'show_fragmentation' not in st.session_state:
        st.session_state.show_fragmentation = False
    if 'use_interactive_view' not in st.session_state:
        st.session_state.use_interactive_view = True  # Default to interactive Plotly view
    # Phase 4: Filtering and zoom options
    if 'filter_materials' not in st.session_state:
        st.session_state.filter_materials = []  # Empty = show all
    if 'filter_row_range' not in st.session_state:
        st.session_state.filter_row_range = None  # (start, end) or None for all
    if 'filter_col_range' not in st.session_state:
        st.session_state.filter_col_range = None  # (start, end) or None for all
    if 'filter_hide_empty_rows' not in st.session_state:
        st.session_state.filter_hide_empty_rows = False
    if 'filter_hide_empty_cols' not in st.session_state:
        st.session_state.filter_hide_empty_cols = False
    if 'focus_mode' not in st.session_state:
        st.session_state.focus_mode = False
    # Undo/Redo history
    if 'undo_stack' not in st.session_state:
        st.session_state.undo_stack = []  # List of (row_order, col_order) tuples
    if 'redo_stack' not in st.session_state:
        st.session_state.redo_stack = []
    if 'max_undo_steps' not in st.session_state:
        st.session_state.max_undo_steps = 20
    # Performance optimization: Edit mode and deferred updates
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False  # When True, matrix visualization is hidden
    if 'pending_metadata_changes' not in st.session_state:
        st.session_state.pending_metadata_changes = {}  # Store changes until Apply is clicked
    if 'matrix_needs_refresh' not in st.session_state:
        st.session_state.matrix_needs_refresh = True  # Flag to control when matrix is redrawn
    if 'cached_matrix_fig' not in st.session_state:
        st.session_state.cached_matrix_fig = None  # Cache the Plotly figure
    if 'cached_matrix_config' not in st.session_state:
        st.session_state.cached_matrix_config = None
    if 'last_viz_params' not in st.session_state:
        st.session_state.last_viz_params = {}  # Track params to detect changes
    # Export caching for reliable downloads
    if 'export_image_data' not in st.session_state:
        st.session_state.export_image_data = None
    if 'export_image_format' not in st.session_state:
        st.session_state.export_image_format = None
    if 'export_image_filename' not in st.session_state:
        st.session_state.export_image_filename = None
    if 'export_legend_data' not in st.session_state:
        st.session_state.export_legend_data = None
    # Data export caching
    if 'export_data_content' not in st.session_state:
        st.session_state.export_data_content = None
    if 'export_data_filename' not in st.session_state:
        st.session_state.export_data_filename = None
    if 'export_data_mime' not in st.session_state:
        st.session_state.export_data_mime = None
    if 'export_data_type' not in st.session_state:
        st.session_state.export_data_type = None
    # Biplot export caching
    if 'export_biplot_data' not in st.session_state:
        st.session_state.export_biplot_data = None
    # Counter for unique download button keys
    if 'export_counter' not in st.session_state:
        st.session_state.export_counter = 0


def save_state_for_undo():
    """Save current state to undo stack before making changes"""
    if st.session_state.row_order and st.session_state.col_order:
        current_state = (
            list(st.session_state.row_order),
            list(st.session_state.col_order)
        )
        st.session_state.undo_stack.append(current_state)
        # Limit stack size
        if len(st.session_state.undo_stack) > st.session_state.max_undo_steps:
            st.session_state.undo_stack.pop(0)
        # Clear redo stack when new action is performed
        st.session_state.redo_stack = []


def undo():
    """Undo last change"""
    if st.session_state.undo_stack:
        # Save current state to redo stack
        current_state = (
            list(st.session_state.row_order),
            list(st.session_state.col_order)
        )
        st.session_state.redo_stack.append(current_state)
        # Restore previous state
        prev_row_order, prev_col_order = st.session_state.undo_stack.pop()
        st.session_state.row_order = prev_row_order
        st.session_state.col_order = prev_col_order
        return True
    return False


def redo():
    """Redo last undone change"""
    if st.session_state.redo_stack:
        # Save current state to undo stack
        current_state = (
            list(st.session_state.row_order),
            list(st.session_state.col_order)
        )
        st.session_state.undo_stack.append(current_state)
        # Restore redo state
        redo_row_order, redo_col_order = st.session_state.redo_stack.pop()
        st.session_state.row_order = redo_row_order
        st.session_state.col_order = redo_col_order
        return True
    return False


# ============================================================
# DATA IMPORT / EXPORT FUNCTIONS
# ============================================================

def load_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV file and return DataFrame"""
    df = pd.read_csv(uploaded_file, index_col=0)
    return df


def load_excel(uploaded_file) -> pd.DataFrame:
    """Load Excel file and return DataFrame"""
    df = pd.read_excel(uploaded_file, index_col=0)
    return df


def detect_data_type(df: pd.DataFrame) -> str:
    """Detect if data is presence/absence or frequency"""
    values = df.values.flatten()
    unique_values = set(values[~pd.isna(values)])
    if unique_values.issubset({0, 1, 0.0, 1.0}):
        return "presence_absence"
    return "frequency"


def initialize_metadata(project: Project):
    """Initialize metadata for all rows and columns"""
    if project.matrix is not None:
        for col in project.matrix.columns:
            if col not in project.column_metadata:
                project.column_metadata[col] = asdict(ColumnMetadata(name=col))
        for row in project.matrix.index:
            if row not in project.row_metadata:
                project.row_metadata[row] = asdict(RowMetadata(name=row))


def export_project(project: Project) -> str:
    """Export project to JSON string including visualization settings"""
    export_data = {
        'name': project.name,
        'data_type': project.data_type,
        'matrix': project.matrix.to_dict() if project.matrix is not None else None,
        'matrix_index': list(project.matrix.index) if project.matrix is not None else [],
        'column_metadata': project.column_metadata,
        'row_metadata': project.row_metadata,
        'cell_annotations': project.cell_annotations,
        'material_groups': project.material_groups,
        'context_types': project.context_types,
        'row_order': st.session_state.row_order,
        'col_order': st.session_state.col_order,
        # Visualization settings
        'visualization_settings': {
            'viz_style': st.session_state.get('viz_style', 'classic'),
            'cell_size': st.session_state.get('cell_size', 0.4),
            'show_values': st.session_state.get('show_values', False),
            'show_colors': st.session_state.get('show_colors', True),
            'show_certainty': st.session_state.get('show_certainty', True),
            'show_fragmentation': st.session_state.get('show_fragmentation', False),
        },
        # Filter settings
        'filter_settings': {
            'filter_materials': st.session_state.get('filter_materials', []),
            'filter_row_range': st.session_state.get('filter_row_range', None),
            'filter_col_range': st.session_state.get('filter_col_range', None),
            'filter_hide_empty_rows': st.session_state.get('filter_hide_empty_rows', False),
            'filter_hide_empty_cols': st.session_state.get('filter_hide_empty_cols', False),
        }
    }
    return json.dumps(export_data, indent=2)


def import_project(json_str: str) -> Project:
    """Import project from JSON string including visualization settings"""
    data = json.loads(json_str)
    project = Project(
        name=data.get('name', 'Imported Project'),
        data_type=data.get('data_type', 'presence_absence'),
        column_metadata=data.get('column_metadata', {}),
        row_metadata=data.get('row_metadata', {}),
        cell_annotations=data.get('cell_annotations', {}),
        material_groups=data.get('material_groups', Project().material_groups),
        context_types=data.get('context_types', Project().context_types)
    )
    if data.get('matrix'):
        project.matrix = pd.DataFrame(data['matrix'])
        if data.get('matrix_index'):
            project.matrix.index = data['matrix_index']
    
    # Restore visualization settings
    viz_settings = data.get('visualization_settings', {})
    if viz_settings:
        st.session_state.viz_style = viz_settings.get('viz_style', 'classic')
        st.session_state.cell_size = viz_settings.get('cell_size', 0.4)
        st.session_state.show_values = viz_settings.get('show_values', False)
        st.session_state.show_colors = viz_settings.get('show_colors', True)
        st.session_state.show_certainty = viz_settings.get('show_certainty', True)
        st.session_state.show_fragmentation = viz_settings.get('show_fragmentation', False)
    
    # Restore filter settings
    filter_settings = data.get('filter_settings', {})
    if filter_settings:
        st.session_state.filter_materials = filter_settings.get('filter_materials', [])
        st.session_state.filter_row_range = filter_settings.get('filter_row_range', None)
        st.session_state.filter_col_range = filter_settings.get('filter_col_range', None)
        st.session_state.filter_hide_empty_rows = filter_settings.get('filter_hide_empty_rows', False)
        st.session_state.filter_hide_empty_cols = filter_settings.get('filter_hide_empty_cols', False)
    
    return project, data.get('row_order', []), data.get('col_order', [])


# ============================================================
# SERIATION ALGORITHMS
# ============================================================

def calculate_row_centroid(matrix: pd.DataFrame, row_idx) -> float:
    """Calculate centroid position for a row"""
    row = matrix.loc[row_idx]
    positions = np.arange(len(row))
    weights = row.values.astype(float)
    if weights.sum() == 0:
        return len(row) / 2
    return np.average(positions, weights=weights)


def calculate_col_centroid(matrix: pd.DataFrame, col_name) -> float:
    """Calculate centroid position for a column"""
    col = matrix[col_name]
    positions = np.arange(len(col))
    weights = col.values.astype(float)
    if weights.sum() == 0:
        return len(col) / 2
    return np.average(positions, weights=weights)


def sort_by_centroid(matrix: pd.DataFrame, row_metadata: dict, col_metadata: dict,
                     sort_rows: bool = True, sort_cols: bool = True) -> tuple:
    """Sort matrix by centroid method, respecting fixed items"""
    new_row_order = list(matrix.index)
    new_col_order = list(matrix.columns)
    
    if sort_cols:
        # Get fixed and unfixed columns
        fixed_cols = [(i, c) for i, c in enumerate(new_col_order) 
                      if col_metadata.get(c, {}).get('is_fixed', False)]
        unfixed_cols = [c for c in new_col_order 
                        if not col_metadata.get(c, {}).get('is_fixed', False)]
        
        # Sort unfixed columns
        col_centroids = [(c, calculate_col_centroid(matrix, c)) for c in unfixed_cols]
        col_centroids.sort(key=lambda x: x[1])
        sorted_unfixed = [c for c, _ in col_centroids]
        
        # Reconstruct order maintaining fixed positions
        new_col_order = []
        unfixed_iter = iter(sorted_unfixed)
        fixed_dict = {i: c for i, c in fixed_cols}
        for i in range(len(matrix.columns)):
            if i in fixed_dict:
                new_col_order.append(fixed_dict[i])
            else:
                new_col_order.append(next(unfixed_iter))
    
    # Reorder matrix columns first for row centroid calculation
    temp_matrix = matrix[new_col_order]
    
    if sort_rows:
        # Get fixed and unfixed rows
        fixed_rows = [(i, r) for i, r in enumerate(new_row_order)
                      if row_metadata.get(r, {}).get('is_fixed', False)]
        unfixed_rows = [r for r in new_row_order
                        if not row_metadata.get(r, {}).get('is_fixed', False)]
        
        # Sort unfixed rows
        row_centroids = [(r, calculate_row_centroid(temp_matrix, r)) for r in unfixed_rows]
        row_centroids.sort(key=lambda x: x[1])
        sorted_unfixed = [r for r, _ in row_centroids]
        
        # Reconstruct order maintaining fixed positions
        new_row_order = []
        unfixed_iter = iter(sorted_unfixed)
        fixed_dict = {i: r for i, r in fixed_rows}
        for i in range(len(matrix.index)):
            if i in fixed_dict:
                new_row_order.append(fixed_dict[i])
            else:
                new_row_order.append(next(unfixed_iter))
    
    return new_row_order, new_col_order


def correspondence_analysis(matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Correspondence Analysis on the data matrix.
    
    Returns:
        row_coords: Row coordinates (contexts) in CA space
        col_coords: Column coordinates (types) in CA space  
        eigenvalues: Eigenvalues (inertia) for each dimension
    """
    # Convert to numpy and ensure float
    data = matrix.values.astype(float)
    
    # Handle zero rows/columns
    row_sums = data.sum(axis=1)
    col_sums = data.sum(axis=0)
    
    # Remove zero rows/columns for calculation
    valid_rows = row_sums > 0
    valid_cols = col_sums > 0
    
    if not valid_rows.all() or not valid_cols.all():
        # Work with reduced matrix
        data_reduced = data[valid_rows][:, valid_cols]
        row_sums = data_reduced.sum(axis=1)
        col_sums = data_reduced.sum(axis=0)
    else:
        data_reduced = data
    
    n = data_reduced.sum()
    
    if n == 0:
        # Return zeros if no data
        return np.zeros(len(matrix)), np.zeros(len(matrix.columns)), np.array([0])
    
    # Correspondence matrix
    P = data_reduced / n
    
    # Row and column masses
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    
    # Diagonal matrices of masses
    Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r + 1e-10))
    Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c + 1e-10))
    
    # Standardized residuals matrix
    expected = np.outer(r, c)
    S = Dr_inv_sqrt @ (P - expected) @ Dc_inv_sqrt
    
    # SVD
    try:
        U, sigma, Vt = linalg.svd(S, full_matrices=False)
    except:
        # Fallback if SVD fails
        return np.zeros(len(matrix)), np.zeros(len(matrix.columns)), np.array([0])
    
    # Standard coordinates
    row_coords_reduced = Dr_inv_sqrt @ U
    col_coords_reduced = Dc_inv_sqrt @ Vt.T
    
    # Eigenvalues (squared singular values = inertia)
    eigenvalues = sigma ** 2
    
    # Map back to full matrix if needed
    if not valid_rows.all() or not valid_cols.all():
        row_coords = np.zeros((len(matrix), row_coords_reduced.shape[1]))
        col_coords = np.zeros((len(matrix.columns), col_coords_reduced.shape[1]))
        row_coords[valid_rows] = row_coords_reduced
        col_coords[valid_cols] = col_coords_reduced
    else:
        row_coords = row_coords_reduced
        col_coords = col_coords_reduced
    
    return row_coords, col_coords, eigenvalues


def sort_by_ca(matrix: pd.DataFrame, row_metadata: dict, col_metadata: dict,
               sort_rows: bool = True, sort_cols: bool = True,
               dimension: int = 0) -> Tuple[list, list, dict]:
    """
    Sort matrix using Correspondence Analysis first dimension scores.
    
    Returns:
        new_row_order: Sorted row indices
        new_col_order: Sorted column indices
        ca_info: Dict with CA results for visualization
    """
    row_coords, col_coords, eigenvalues = correspondence_analysis(matrix)
    
    new_row_order = list(matrix.index)
    new_col_order = list(matrix.columns)
    
    # Calculate total inertia and explained variance
    total_inertia = eigenvalues.sum()
    if total_inertia > 0:
        explained_variance = eigenvalues / total_inertia * 100
    else:
        explained_variance = np.zeros_like(eigenvalues)
    
    if sort_cols and col_coords.shape[1] > dimension:
        # Get fixed and unfixed columns
        fixed_cols = [(i, c) for i, c in enumerate(new_col_order)
                      if col_metadata.get(c, {}).get('is_fixed', False)]
        unfixed_cols = [c for c in new_col_order
                        if not col_metadata.get(c, {}).get('is_fixed', False)]
        
        # Sort unfixed by CA score
        col_scores = {c: col_coords[list(matrix.columns).index(c), dimension] 
                      for c in unfixed_cols}
        sorted_unfixed = sorted(unfixed_cols, key=lambda c: col_scores[c])
        
        # Reconstruct order
        new_col_order = []
        unfixed_iter = iter(sorted_unfixed)
        fixed_dict = {i: c for i, c in fixed_cols}
        for i in range(len(matrix.columns)):
            if i in fixed_dict:
                new_col_order.append(fixed_dict[i])
            else:
                new_col_order.append(next(unfixed_iter))
    
    if sort_rows and row_coords.shape[1] > dimension:
        # Get fixed and unfixed rows
        fixed_rows = [(i, r) for i, r in enumerate(new_row_order)
                      if row_metadata.get(r, {}).get('is_fixed', False)]
        unfixed_rows = [r for r in new_row_order
                        if not row_metadata.get(r, {}).get('is_fixed', False)]
        
        # Sort unfixed by CA score
        row_scores = {r: row_coords[list(matrix.index).index(r), dimension]
                      for r in unfixed_rows}
        sorted_unfixed = sorted(unfixed_rows, key=lambda r: row_scores[r])
        
        # Reconstruct order
        new_row_order = []
        unfixed_iter = iter(sorted_unfixed)
        fixed_dict = {i: r for i, r in fixed_rows}
        for i in range(len(matrix.index)):
            if i in fixed_dict:
                new_row_order.append(fixed_dict[i])
            else:
                new_row_order.append(next(unfixed_iter))
    
    ca_info = {
        'row_coords': row_coords,
        'col_coords': col_coords,
        'eigenvalues': eigenvalues,
        'explained_variance': explained_variance,
        'total_inertia': total_inertia,
        'row_names': list(matrix.index),
        'col_names': list(matrix.columns)
    }
    
    return new_row_order, new_col_order, ca_info


def calculate_seriation_quality(matrix: pd.DataFrame, row_order: list, col_order: list) -> dict:
    """
    Calculate quality metrics for the current seriation.
    
    Metrics:
    - concentration: How concentrated are values along the diagonal?
    - anti_robinson: Measure of anti-Robinson form (lower = better)
    - gaps: Number and size of gaps in type distributions
    """
    ordered = matrix.loc[row_order, col_order].values.astype(float)
    n_rows, n_cols = ordered.shape
    
    # 1. Concentration Index (CI)
    # Measures how close non-zero values are to the diagonal
    total_weight = 0
    weighted_distance = 0
    
    for i in range(n_rows):
        for j in range(n_cols):
            if ordered[i, j] > 0:
                # Normalized position
                row_pos = i / max(1, n_rows - 1)
                col_pos = j / max(1, n_cols - 1)
                # Distance from diagonal
                diag_dist = abs(row_pos - col_pos)
                weighted_distance += ordered[i, j] * diag_dist
                total_weight += ordered[i, j]
    
    if total_weight > 0:
        concentration = 1 - (weighted_distance / total_weight)
    else:
        concentration = 0
    
    # 2. Anti-Robinson Index
    # Count violations of Robinson property (values should decrease away from diagonal)
    violations = 0
    total_comparisons = 0
    
    for j in range(n_cols):
        col = ordered[:, j]
        nonzero_indices = np.where(col > 0)[0]
        if len(nonzero_indices) >= 2:
            first, last = nonzero_indices[0], nonzero_indices[-1]
            # Check for zeros within the range (gaps)
            for i in range(first, last + 1):
                if col[i] == 0:
                    violations += 1
                total_comparisons += 1
    
    if total_comparisons > 0:
        anti_robinson = 1 - (violations / total_comparisons)
    else:
        anti_robinson = 1
    
    # 3. Gap Analysis
    gaps = []
    for j, col_name in enumerate(col_order):
        col = ordered[:, j]
        nonzero_indices = np.where(col > 0)[0]
        if len(nonzero_indices) >= 2:
            first, last = nonzero_indices[0], nonzero_indices[-1]
            gap_count = sum(1 for i in range(first, last + 1) if col[i] == 0)
            if gap_count > 0:
                gaps.append({
                    'column': col_name,
                    'gaps': gap_count,
                    'span': last - first + 1
                })
    
    # 4. Type Span Continuity
    spans = []
    for j, col_name in enumerate(col_order):
        col = ordered[:, j]
        nonzero_indices = np.where(col > 0)[0]
        if len(nonzero_indices) >= 1:
            first, last = nonzero_indices[0], nonzero_indices[-1]
            span = last - first + 1
            actual_count = len(nonzero_indices)
            continuity = actual_count / span if span > 0 else 1
            spans.append({
                'column': col_name,
                'first': first,
                'last': last,
                'span': span,
                'count': actual_count,
                'continuity': continuity
            })
    
    avg_continuity = np.mean([s['continuity'] for s in spans]) if spans else 0
    
    # 5. Overall Quality Score (weighted combination)
    quality_score = (concentration * 0.4 + anti_robinson * 0.4 + avg_continuity * 0.2)
    
    return {
        'concentration': concentration,
        'anti_robinson': anti_robinson,
        'avg_continuity': avg_continuity,
        'quality_score': quality_score,
        'gaps': gaps,
        'spans': spans,
        'n_gaps': len(gaps),
        'total_gap_cells': sum(g['gaps'] for g in gaps)
    }


def iterative_seriation(matrix: pd.DataFrame, row_metadata: dict, col_metadata: dict,
                        max_iterations: int = 10, method: str = 'centroid') -> Tuple[list, list, list]:
    """
    Perform iterative seriation until convergence or max iterations.
    
    Returns:
        final_row_order
        final_col_order
        history: List of quality scores per iteration
    """
    current_row_order = list(matrix.index)
    current_col_order = list(matrix.columns)
    history = []
    
    for iteration in range(max_iterations):
        # Calculate current quality
        quality = calculate_seriation_quality(matrix, current_row_order, current_col_order)
        history.append({
            'iteration': iteration,
            'quality_score': quality['quality_score'],
            'concentration': quality['concentration'],
            'anti_robinson': quality['anti_robinson']
        })
        
        # Sort
        if method == 'centroid':
            new_row_order, new_col_order = sort_by_centroid(
                matrix.loc[current_row_order, current_col_order],
                row_metadata, col_metadata
            )
            # Map back to original names
            new_row_order = [current_row_order[list(matrix.loc[current_row_order].index).index(r)] 
                           if r in matrix.loc[current_row_order].index else r for r in new_row_order]
        elif method == 'ca':
            new_row_order, new_col_order, _ = sort_by_ca(
                matrix, row_metadata, col_metadata
            )
        else:
            break
        
        # Check for convergence
        if new_row_order == current_row_order and new_col_order == current_col_order:
            break
        
        current_row_order = new_row_order
        current_col_order = new_col_order
    
    # Final quality
    quality = calculate_seriation_quality(matrix, current_row_order, current_col_order)
    history.append({
        'iteration': len(history),
        'quality_score': quality['quality_score'],
        'concentration': quality['concentration'],
        'anti_robinson': quality['anti_robinson']
    })
    
    return current_row_order, current_col_order, history


def create_ca_biplot(ca_info: dict, project, figsize: Tuple[int, int] = (10, 8),
                     show_row_labels: bool = True, show_col_labels: bool = True,
                     label_font_size: int = 8, point_size: int = 80,
                     arrow_width: float = 1.5, use_adjusttext: bool = True) -> plt.Figure:
    """Create an improved CA biplot with better label placement and visuals."""
    
    row_coords = ca_info['row_coords']
    col_coords = ca_info['col_coords']
    eigenvalues = ca_info['eigenvalues']
    explained = ca_info['explained_variance']
    row_names = ca_info['row_names']
    col_names = ca_info['col_names']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set a clean style
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Store text objects for potential adjustment
    texts = []
    
    # Plot rows (contexts) as points
    if row_coords.shape[1] >= 2:
        # Get context types for coloring
        context_colors = []
        context_markers = []
        marker_map = {'Grave': 'o', 'Pit': 's', 'Ditch': '^', 'Layer': 'D', 
                      'Posthole': 'p', 'Structure': 'H', 'Unassigned': 'o'}
        
        for name in row_names:
            row_meta = project.row_metadata.get(name, {})
            ctx_type = row_meta.get('context_type', 'Unassigned')
            context_markers.append(marker_map.get(ctx_type, 'o'))
        
        # Plot each context point
        for i, name in enumerate(row_names):
            ax.scatter(row_coords[i, 0], row_coords[i, 1], 
                      c='steelblue', s=point_size, alpha=0.7, 
                      marker=context_markers[i], edgecolors='white', linewidths=0.5,
                      zorder=3)
        
        # Add row labels
        if show_row_labels:
            for i, name in enumerate(row_names):
                display_name = name.replace('_', ' ')
                # Truncate long names
                if len(display_name) > 15:
                    display_name = display_name[:12] + '...'
                txt = ax.annotate(display_name, 
                           (row_coords[i, 0], row_coords[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=label_font_size, color='steelblue',
                           alpha=0.9, fontweight='medium',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                    edgecolor='none', alpha=0.7))
                texts.append(txt)
    
    # Plot columns (types) as arrows with material group colors
    if col_coords.shape[1] >= 2:
        # Calculate scaling factor for arrows (make them visible but not overwhelming)
        max_row_dist = np.max(np.sqrt(row_coords[:, 0]**2 + row_coords[:, 1]**2)) if len(row_coords) > 0 else 1
        max_col_dist = np.max(np.sqrt(col_coords[:, 0]**2 + col_coords[:, 1]**2)) if len(col_coords) > 0 else 1
        
        if max_col_dist > 0:
            scale_factor = max_row_dist / max_col_dist * 0.8
        else:
            scale_factor = 1
        
        for i, name in enumerate(col_names):
            col_meta = project.column_metadata.get(name, {})
            material = col_meta.get('material_group', 'Unassigned')
            color = project.material_groups.get(material, '#CD853F')
            
            # Scale coordinates
            x_end = col_coords[i, 0] * scale_factor
            y_end = col_coords[i, 1] * scale_factor
            
            # Skip very short arrows (near origin)
            arrow_length = np.sqrt(x_end**2 + y_end**2)
            if arrow_length < 0.01:
                continue
            
            # Draw arrow
            ax.annotate('', xy=(x_end, y_end), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color=color, 
                                      lw=arrow_width, alpha=0.7,
                                      shrinkA=0, shrinkB=0))
            
            # Add label at arrow tip
            if show_col_labels:
                display_name = name.replace('_', ' ')
                if len(display_name) > 12:
                    display_name = display_name[:10] + '...'
                
                # Position label slightly beyond arrow tip
                label_x = x_end * 1.08
                label_y = y_end * 1.08
                
                # Determine text alignment based on position
                ha = 'left' if label_x >= 0 else 'right'
                va = 'bottom' if label_y >= 0 else 'top'
                
                txt = ax.annotate(display_name,
                           (label_x, label_y),
                           fontsize=label_font_size - 1, color=color, 
                           ha=ha, va=va, alpha=0.9, fontweight='medium',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                    edgecolor='none', alpha=0.6))
                texts.append(txt)
    
    # Try to adjust text positions to avoid overlap (simple repulsion algorithm)
    if use_adjusttext and len(texts) > 1:
        try:
            # Simple iterative repulsion for overlapping labels
            for iteration in range(50):
                moved = False
                for i, txt1 in enumerate(texts):
                    bbox1 = txt1.get_window_extent(renderer=fig.canvas.get_renderer())
                    for j, txt2 in enumerate(texts):
                        if i >= j:
                            continue
                        bbox2 = txt2.get_window_extent(renderer=fig.canvas.get_renderer())
                        
                        # Check overlap
                        if bbox1.overlaps(bbox2):
                            # Get current positions
                            pos1 = txt1.get_position()
                            pos2 = txt2.get_position()
                            
                            # Calculate repulsion vector
                            dx = pos2[0] - pos1[0]
                            dy = pos2[1] - pos1[1]
                            dist = np.sqrt(dx**2 + dy**2)
                            
                            if dist > 0:
                                # Move texts apart
                                shift = 0.02 * max_row_dist
                                txt1.set_position((pos1[0] - dx/dist * shift, pos1[1] - dy/dist * shift))
                                txt2.set_position((pos2[0] + dx/dist * shift, pos2[1] + dy/dist * shift))
                                moved = True
                
                if not moved:
                    break
        except:
            pass  # If adjustment fails, continue with original positions
    
    # Add axes through origin
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=1)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Labels
    if len(explained) >= 2:
        ax.set_xlabel(f'Dimension 1 ({explained[0]:.1f}% inertia)', fontsize=11, fontweight='medium')
        ax.set_ylabel(f'Dimension 2 ({explained[1]:.1f}% inertia)', fontsize=11, fontweight='medium')
    else:
        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
    
    ax.set_title('Correspondence Analysis Biplot', fontsize=14, fontweight='bold', pad=15)
    
    # Create custom legend
    legend_elements = [
        plt.scatter([], [], c='steelblue', s=60, label='Contexts', alpha=0.7, edgecolors='white'),
    ]
    
    # Add material groups to legend
    materials_in_use = set()
    for name in col_names:
        col_meta = project.column_metadata.get(name, {})
        materials_in_use.add(col_meta.get('material_group', 'Unassigned'))
    
    for material in materials_in_use:
        color = project.material_groups.get(material, '#808080')
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linewidth=2, label=f'{material}')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
              framealpha=0.9, edgecolor='lightgray')
    
    # Equal aspect ratio for proper geometric interpretation
    ax.set_aspect('equal', adjustable='datalim')
    
    # Add some padding to the limits
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    x_pad = (x_range[1] - x_range[0]) * 0.1
    y_pad = (y_range[1] - y_range[0]) * 0.1
    ax.set_xlim(x_range[0] - x_pad, x_range[1] + x_pad)
    ax.set_ylim(y_range[0] - y_pad, y_range[1] + y_pad)
    
    plt.tight_layout()
    return fig


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_matrix_figure(matrix: pd.DataFrame, project: Project,
                         row_order: list, col_order: list,
                         cell_size: float = 0.4,
                         show_values: bool = False,
                         show_material_colors: bool = True,
                         show_certainty: bool = True,
                         show_fragmentation: bool = False,
                         visualization_style: str = "classic") -> plt.Figure:
    """Create publication-ready matrix visualization
    
    visualization_style options:
    - 'classic': filled squares
    - 'battleship': curved frequency bars
    - 'dots': sized dots for frequency
    """
    
    # Reorder matrix
    ordered_matrix = matrix.loc[row_order, col_order]
    n_rows, n_cols = ordered_matrix.shape
    
    # Calculate figure size
    fig_width = max(8, n_cols * cell_size + 4)
    fig_height = max(6, n_rows * cell_size + 3)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Safety check for empty matrix
    if n_rows == 0 or n_cols == 0:
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    # Get max value for frequency scaling
    matrix_values = ordered_matrix.values
    if matrix_values.size > 0 and matrix_values.max() > 0:
        max_val = matrix_values.max()
    else:
        max_val = 1
    
    # Draw cells
    for i, row_idx in enumerate(row_order):
        for j, col_name in enumerate(col_order):
            value = ordered_matrix.loc[row_idx, col_name]
            
            # Get cell annotation if exists
            cell_key = f"{row_idx}_{col_name}"
            annotation = project.cell_annotations.get(cell_key, {})
            certainty = annotation.get('certainty', 'certain')
            fragmentation = annotation.get('fragmentation', 'unknown')
            
            # Get certainty style
            cert_style = CERTAINTY_STYLES.get(certainty, CERTAINTY_STYLES['certain'])
            
            # Cell position (bottom-left corner)
            x_pos = j
            y_pos = n_rows - i - 1
            
            if visualization_style == "classic":
                # Classic filled squares
                if project.data_type == "presence_absence":
                    if value == 1:
                        facecolor = 'black'
                        alpha = cert_style['alpha'] if show_certainty else 1.0
                        hatch = cert_style['hatch'] if show_certainty else None
                    else:
                        facecolor = 'white'
                        alpha = 1.0
                        hatch = None
                else:  # frequency
                    if value > 0:
                        intensity = value / max_val
                        facecolor = plt.cm.Greys(intensity * 0.8 + 0.1)
                        alpha = cert_style['alpha'] if show_certainty else 1.0
                        hatch = cert_style['hatch'] if show_certainty else None
                    else:
                        facecolor = 'white'
                        alpha = 1.0
                        hatch = None
                
                # Draw cell background
                rect = mpatches.Rectangle(
                    (x_pos, y_pos), 1, 1,
                    facecolor=facecolor,
                    edgecolor='darkgray',
                    linewidth=0.5,
                    alpha=alpha,
                    hatch=hatch
                )
                ax.add_patch(rect)
                
                # Show fragmentation marker
                if show_fragmentation and value > 0 and fragmentation != 'unknown':
                    marker_style = FRAGMENTATION_MARKERS.get(fragmentation, {})
                    marker = marker_style.get('marker', 'o')
                    marker_color = 'white' if value > 0 else 'gray'
                    ax.plot(x_pos + 0.5, y_pos + 0.5, marker=marker, 
                           markersize=cell_size * 8, color=marker_color,
                           markeredgecolor='darkgray', markeredgewidth=0.5)
                
                # Show value for frequency data
                if show_values and project.data_type == "frequency" and value > 0:
                    text_color = 'white' if value / max_val > 0.5 else 'black'
                    ax.text(x_pos + 0.5, y_pos + 0.5, str(int(value)),
                           ha='center', va='center', fontsize=7, color=text_color,
                           fontweight='bold')
            
            elif visualization_style == "battleship":
                # Battleship curves - horizontal bars proportional to frequency
                # Draw cell border
                rect = mpatches.Rectangle(
                    (x_pos, y_pos), 1, 1,
                    facecolor='white',
                    edgecolor='lightgray',
                    linewidth=0.3
                )
                ax.add_patch(rect)
                
                if value > 0:
                    # Calculate bar height proportional to value
                    if project.data_type == "presence_absence":
                        bar_height = 0.8
                    else:
                        bar_height = (value / max_val) * 0.9
                    
                    alpha = cert_style['alpha'] if show_certainty else 1.0
                    
                    # Draw centered bar
                    bar_bottom = y_pos + (1 - bar_height) / 2
                    bar_rect = mpatches.Rectangle(
                        (x_pos + 0.1, bar_bottom), 0.8, bar_height,
                        facecolor='black',
                        alpha=alpha,
                        edgecolor='none'
                    )
                    ax.add_patch(bar_rect)
            
            elif visualization_style == "dots":
                # Sized dots for frequency
                # Draw cell border
                rect = mpatches.Rectangle(
                    (x_pos, y_pos), 1, 1,
                    facecolor='white',
                    edgecolor='lightgray',
                    linewidth=0.3
                )
                ax.add_patch(rect)
                
                if value > 0:
                    if project.data_type == "presence_absence":
                        dot_size = cell_size * 15
                    else:
                        dot_size = (value / max_val) * cell_size * 20 + 2
                    
                    alpha = cert_style['alpha'] if show_certainty else 1.0
                    
                    ax.plot(x_pos + 0.5, y_pos + 0.5, 'o',
                           markersize=dot_size, color='black', alpha=alpha)
    
    # Draw material group colors in header
    if show_material_colors:
        for j, col_name in enumerate(col_order):
            col_meta = project.column_metadata.get(col_name, {})
            material = col_meta.get('material_group', 'Unassigned')
            color = project.material_groups.get(material, '#808080')
            
            # Color band
            rect = mpatches.Rectangle(
                (j, n_rows), 1, 0.4,
                facecolor=color,
                edgecolor='darkgray',
                linewidth=0.5
            )
            ax.add_patch(rect)
            
            # Mark index types with a star
            if col_meta.get('is_index_type', False):
                ax.plot(j + 0.5, n_rows + 0.2, '*', markersize=8, 
                       color='gold', markeredgecolor='black', markeredgewidth=0.5)
    
    # Mark fixed rows/columns
    for i, row_idx in enumerate(row_order):
        row_meta = project.row_metadata.get(row_idx, {})
        if row_meta.get('is_fixed', False):
            ax.plot(-0.3, n_rows - i - 0.5, marker='D', markersize=6, color='red',
                   markeredgecolor='darkred', markeredgewidth=0.5)
    
    for j, col_name in enumerate(col_order):
        col_meta = project.column_metadata.get(col_name, {})
        if col_meta.get('is_fixed', False):
            y_marker = n_rows + 0.6 if show_material_colors else n_rows + 0.2
            ax.plot(j + 0.5, y_marker, marker='D', markersize=6, color='red',
                   markeredgecolor='darkred', markeredgewidth=0.5)
    
    # Set axis properties
    header_height = 0.4 if show_material_colors else 0
    ax.set_xlim(-0.5, n_cols)
    ax.set_ylim(-0.5, n_rows + header_height + 0.3)
    ax.set_aspect('equal')
    
    # Column labels (rotated)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    col_labels = []
    for col in col_order:
        col_meta = project.column_metadata.get(col, {})
        label = col.replace('_', ' ')
        if col_meta.get('is_index_type', False):
            label = f"★ {label}"
        col_labels.append(label)
    ax.set_xticklabels(col_labels, rotation=90, ha='center', fontsize=8)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # Row labels with context type
    ax.set_yticks([n_rows - i - 0.5 for i in range(n_rows)])
    row_labels = []
    for row in row_order:
        row_meta = project.row_metadata.get(row, {})
        label = row.replace('_', ' ')
        ctx_type = row_meta.get('context_type', '')
        if ctx_type and ctx_type != 'Unassigned':
            label = f"{label} [{ctx_type[:3]}]"
        row_labels.append(label)
    ax.set_yticklabels(row_labels, fontsize=8)
    
    # Remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(length=0)
    
    plt.tight_layout()
    return fig


def create_interactive_matrix_figure(matrix: pd.DataFrame, project: Project,
                                      row_order: list, col_order: list,
                                      cell_size: float = 0.4,
                                      show_values: bool = False,
                                      show_material_colors: bool = True,
                                      show_certainty: bool = True,
                                      visualization_style: str = "classic") -> go.Figure:
    """Create interactive matrix visualization with Plotly.
    
    Features: Zoom-in, Zoom-out, Pan, Reset View (Home), Box Select
    
    Orientation: First row at TOP (like Matplotlib version)
    - row_order[0] appears at top
    - row_order[-1] appears at bottom
    """
    
    # Reorder matrix
    ordered_matrix = matrix.loc[row_order, col_order]
    n_rows, n_cols = ordered_matrix.shape
    
    # Safety check for empty matrix
    if n_rows == 0 or n_cols == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", x=0.5, y=0.5, 
                          showarrow=False, font=dict(size=14))
        return fig, {}
    
    # Get max value for frequency scaling
    matrix_values = ordered_matrix.values
    max_val = matrix_values.max() if matrix_values.size > 0 and matrix_values.max() > 0 else 1
    
    # Prepare data for heatmap - NO reversal needed, we use yaxis autorange='reversed'
    z_data = ordered_matrix.values
    
    # Build cell text and colors
    customdata = []
    
    for i, row_idx in enumerate(row_order):
        row_customdata = []
        for j, col_name in enumerate(col_order):
            value = ordered_matrix.loc[row_idx, col_name]
            
            # Get cell annotation
            cell_key = f"{row_idx}_{col_name}"
            annotation = project.cell_annotations.get(cell_key, {})
            certainty = annotation.get('certainty', 'certain')
            fragmentation = annotation.get('fragmentation', 'unknown')
            notes = annotation.get('notes', '')
            inventory = annotation.get('inventory_numbers', '')
            
            # Get material info
            col_meta = project.column_metadata.get(col_name, {})
            material = col_meta.get('material_group', 'Unassigned')
            
            # Build hover text
            hover_parts = [
                f"<b>Context:</b> {row_idx}",
                f"<b>Type:</b> {col_name}",
                f"<b>Value:</b> {value}",
                f"<b>Material:</b> {material}"
            ]
            if certainty != 'certain':
                hover_parts.append(f"<b>Certainty:</b> {certainty}")
            if fragmentation != 'unknown':
                hover_parts.append(f"<b>Fragmentation:</b> {fragmentation}")
            if inventory:
                hover_parts.append(f"<b>Inventory:</b> {inventory}")
            if notes:
                hover_parts.append(f"<b>Notes:</b> {notes}")
            
            hover_text = "<br>".join(hover_parts)
            row_customdata.append(hover_text)
        
        customdata.append(row_customdata)
    
    # Create heatmap
    fig = go.Figure()
    
    # Main heatmap - use row_order directly as y labels
    fig.add_trace(go.Heatmap(
        z=z_data,
        x=col_order,
        y=row_order,
        customdata=customdata,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=[[0, 'white'], [1, 'black']],
        showscale=False,
        xgap=1,
        ygap=1
    ))
    
    # Prepare display labels
    col_display_labels = []
    for col in col_order:
        col_meta = project.column_metadata.get(col, {})
        label = col.replace('_', ' ')
        if col_meta.get('is_index_type', False):
            label = f"★ {label}"
        col_display_labels.append(label)
    
    row_display_labels = []
    for row in row_order:
        row_meta = project.row_metadata.get(row, {})
        label = row.replace('_', ' ')
        ctx_type = row_meta.get('context_type', '')
        if ctx_type and ctx_type != 'Unassigned':
            label = f"{label} [{ctx_type[:3]}]"
        row_display_labels.append(label)
    
    # Calculate figure size
    base_cell_px = 25 * cell_size / 0.4  # Scale based on cell_size
    fig_width = max(600, n_cols * base_cell_px + 200)
    fig_height = max(400, n_rows * base_cell_px + 200)  # More height for title
    
    # Update layout - key: autorange='reversed' puts first row at TOP
    fig.update_layout(
        title=dict(
            text=f"{project.name} - Combination Matrix",
            font=dict(size=16, color='#8B4513'),
            y=0.98,  # Position title at very top
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=col_order,
            ticktext=col_display_labels,
            tickangle=-90,
            side='top',
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=row_order,
            ticktext=row_display_labels,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            autorange='reversed'  # This puts row_order[0] at TOP
        ),
        width=min(fig_width, 1400),
        height=min(fig_height, 900),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=120, r=50, t=220, b=50),  # Increased top margin for title + labels + material bar
        # Enable modebar with zoom tools
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.8)',
            color='#666',
            activecolor='#8B4513'
        )
    )
    
    # Add material color bar ABOVE the column labels using scatter markers
    if show_material_colors:
        # Collect colors and hover info for each column
        marker_colors = []
        hover_texts = []
        index_type_cols = []
        
        for j, col_name in enumerate(col_order):
            col_meta = project.column_metadata.get(col_name, {})
            material = col_meta.get('material_group', 'Unassigned')
            color = project.material_groups.get(material, '#808080')
            marker_colors.append(color)
            hover_texts.append(f"{col_name}<br>Material: {material}")
            
            if col_meta.get('is_index_type', False):
                index_type_cols.append(col_name)
        
        # Add material bar as a scatter trace with square markers
        fig.add_trace(go.Scatter(
            x=col_order,
            y=[row_order[0]] * n_cols,  # Position at first row (which is at top due to reversed axis)
            mode='markers',
            marker=dict(
                symbol='square',
                size=15,
                color=marker_colors,
                line=dict(color='darkgray', width=1)
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False,
            yaxis='y2'  # Use secondary y-axis for positioning
        ))
        
        # Add secondary y-axis for material bar (positioned above main plot)
        fig.update_layout(
            yaxis2=dict(
                overlaying='y',
                side='top',
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[0, 1],
                fixedrange=True,
                domain=[0.92, 0.95]  # Small area at top
            )
        )
        
        # Mark index types with stars
        if index_type_cols:
            fig.add_trace(go.Scatter(
                x=index_type_cols,
                y=[0.5] * len(index_type_cols),
                mode='markers+text',
                marker=dict(
                    symbol='star',
                    size=12,
                    color='gold',
                    line=dict(color='darkgoldenrod', width=1)
                ),
                text=['★'] * len(index_type_cols),
                textposition='top center',
                textfont=dict(size=10, color='gold'),
                hovertext=[f"{c} (Index Type)" for c in index_type_cols],
                hoverinfo='text',
                showlegend=False,
                yaxis='y2'
            ))
    
    # Configure zoom/pan behavior
    fig.update_layout(
        dragmode='pan',  # Default to pan mode
        hovermode='closest'
    )
    
    # Add value annotations if requested (for frequency data)
    if show_values and project.data_type == "frequency":
        for i, row_idx in enumerate(row_order):
            for j, col_name in enumerate(col_order):
                value = ordered_matrix.loc[row_idx, col_name]
                if value > 0:
                    text_color = 'white' if value / max_val > 0.5 else 'black'
                    fig.add_annotation(
                        x=col_name,
                        y=row_idx,
                        text=str(int(value)),
                        showarrow=False,
                        font=dict(size=8, color=text_color, family='Arial Black')
                    )
    
    # Custom modebar buttons config
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d'],
        'modeBarButtonsToAdd': [],
        'scrollZoom': True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{project.name}_matrix',
            'height': 1200,
            'width': 1600,
            'scale': 2
        }
    }
    
    return fig, config


def apply_matrix_filters(row_order: list, col_order: list, matrix: pd.DataFrame, 
                         project: Project) -> Tuple[list, list]:
    """Apply filtering options to row and column orders.
    
    Returns filtered row_order and col_order based on session state filters.
    """
    filtered_rows = list(row_order)
    filtered_cols = list(col_order)
    
    # Filter by material groups
    if st.session_state.filter_materials:
        filtered_cols = [
            col for col in filtered_cols
            if project.column_metadata.get(col, {}).get('material_group', 'Unassigned') 
               in st.session_state.filter_materials
        ]
    
    # Filter by row range (focus mode)
    if st.session_state.filter_row_range:
        start, end = st.session_state.filter_row_range
        filtered_rows = filtered_rows[start:end+1]
    
    # Filter by column range (focus mode)
    if st.session_state.filter_col_range:
        start, end = st.session_state.filter_col_range
        # Apply to already filtered cols
        original_positions = [col_order.index(c) for c in filtered_cols]
        filtered_cols = [c for c, pos in zip(filtered_cols, original_positions) 
                        if start <= pos <= end]
    
    # Hide empty rows (rows with no finds in visible columns)
    if st.session_state.filter_hide_empty_rows and filtered_cols:
        filtered_rows = [
            row for row in filtered_rows
            if matrix.loc[row, filtered_cols].sum() > 0
        ]
    
    # Hide empty columns (columns with no finds in visible rows)
    if st.session_state.filter_hide_empty_cols and filtered_rows:
        filtered_cols = [
            col for col in filtered_cols
            if matrix.loc[filtered_rows, col].sum() > 0
        ]
    
    return filtered_rows, filtered_cols


def get_matrix_size_category(n_rows: int, n_cols: int) -> str:
    """Categorize matrix size for adaptive display settings."""
    total_cells = n_rows * n_cols
    if total_cells <= 200:
        return "small"  # Up to ~15x15
    elif total_cells <= 1000:
        return "medium"  # Up to ~30x30
    elif total_cells <= 5000:
        return "large"  # Up to ~70x70
    else:
        return "very_large"  # 100x100+


def get_recommended_settings(size_category: str) -> dict:
    """Get recommended visualization settings based on matrix size."""
    settings = {
        "small": {
            "cell_size": 0.5,
            "show_labels": True,
            "label_size": 8,
            "warning": None
        },
        "medium": {
            "cell_size": 0.35,
            "show_labels": True,
            "label_size": 7,
            "warning": None
        },
        "large": {
            "cell_size": 0.25,
            "show_labels": True,
            "label_size": 6,
            "warning": "Large matrix - consider filtering by material group"
        },
        "very_large": {
            "cell_size": 0.15,
            "show_labels": False,
            "label_size": 5,
            "warning": "Very large matrix - filtering strongly recommended"
        }
    }
    return settings.get(size_category, settings["medium"])


def create_legend_figure(project: Project, show_certainty: bool = True,
                         show_fragmentation: bool = False) -> plt.Figure:
    """Create a comprehensive legend figure"""
    
    # Count legend items
    materials = [m for m in project.material_groups.keys() if m != "Unassigned"]
    n_materials = len(materials)
    n_certainty = 3 if show_certainty else 0
    n_frag = 3 if show_fragmentation else 0
    
    total_items = n_materials + n_certainty + n_frag + 2  # +2 for headers
    
    fig_height = max(2, 0.35 * total_items + 0.5)
    fig, ax = plt.subplots(figsize=(3.5, fig_height))
    
    y_pos = total_items - 0.5
    
    # Material groups section
    ax.text(0, y_pos, "Material Groups", fontsize=9, fontweight='bold')
    y_pos -= 0.8
    
    for material in materials:
        color = project.material_groups[material]
        rect = mpatches.Rectangle(
            (0, y_pos - 0.3), 0.4, 0.5,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(rect)
        ax.text(0.55, y_pos, material, fontsize=8, va='center')
        y_pos -= 0.7
    
    # Certainty section
    if show_certainty:
        y_pos -= 0.3
        ax.text(0, y_pos, "Certainty", fontsize=9, fontweight='bold')
        y_pos -= 0.8
        
        for cert_name, cert_style in CERTAINTY_STYLES.items():
            rect = mpatches.Rectangle(
                (0, y_pos - 0.3), 0.4, 0.5,
                facecolor='black',
                edgecolor='darkgray',
                linewidth=0.5,
                alpha=cert_style['alpha'],
                hatch=cert_style['hatch']
            )
            ax.add_patch(rect)
            ax.text(0.55, y_pos, cert_name.capitalize(), fontsize=8, va='center')
            y_pos -= 0.7
    
    # Fragmentation section
    if show_fragmentation:
        y_pos -= 0.3
        ax.text(0, y_pos, "Fragmentation", fontsize=9, fontweight='bold')
        y_pos -= 0.8
        
        for frag_name, frag_style in FRAGMENTATION_MARKERS.items():
            ax.plot(0.2, y_pos, marker=frag_style['marker'], markersize=10,
                   color='black', markeredgecolor='darkgray')
            ax.text(0.55, y_pos, frag_name.capitalize(), fontsize=8, va='center')
            y_pos -= 0.7
    
    ax.set_xlim(-0.1, 3)
    ax.set_ylim(y_pos - 0.5, total_items)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_summary_statistics(matrix: pd.DataFrame, project: Project,
                              row_order: list, col_order: list) -> pd.DataFrame:
    """Generate summary statistics for the seriation"""
    ordered_matrix = matrix.loc[row_order, col_order]
    
    stats = []
    for col in col_order:
        col_data = ordered_matrix[col]
        presence_rows = col_data[col_data > 0].index.tolist()
        
        if presence_rows:
            first_idx = row_order.index(presence_rows[0])
            last_idx = row_order.index(presence_rows[-1])
            span = last_idx - first_idx + 1
        else:
            first_idx = last_idx = span = 0
        
        col_meta = project.column_metadata.get(col, {})
        
        stats.append({
            'Type': col,
            'Material': col_meta.get('material_group', 'Unassigned'),
            'Index Type': '★' if col_meta.get('is_index_type', False) else '',
            'Count': int(col_data.sum()),
            'Contexts': int((col_data > 0).sum()),
            'First': first_idx + 1 if presence_rows else '-',
            'Last': last_idx + 1 if presence_rows else '-',
            'Span': span
        })
    
    return pd.DataFrame(stats)


# ============================================================
# UI COMPONENTS
# ============================================================

def render_sidebar():
    """Render sidebar with project controls"""
    st.sidebar.markdown("## 🏺 CombiTab")
    st.sidebar.markdown("*Archaeological Seriation Tool*")
    st.sidebar.divider()
    
    # Project name
    st.session_state.project.name = st.sidebar.text_input(
        "Project Name",
        value=st.session_state.project.name
    )
    
    st.sidebar.divider()
    
    # Data Import Section
    st.sidebar.markdown("### 📁 Data Import")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=['csv', 'xlsx', 'xls'],
        help="First column should contain context/row names"
    )
    
    if uploaded_file is not None:
        # Only process if this is a new file (check by name)
        file_key = f"loaded_file_{uploaded_file.name}"
        if file_key not in st.session_state:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = load_csv(uploaded_file)
                else:
                    df = load_excel(uploaded_file)
                
                st.session_state.project.matrix = df
                st.session_state.project.data_type = detect_data_type(df)
                initialize_metadata(st.session_state.project)
                st.session_state.row_order = list(df.index)
                st.session_state.col_order = list(df.columns)
                st.session_state[file_key] = True  # Mark as loaded
                st.sidebar.success(f"Loaded: {len(df)} contexts × {len(df.columns)} types")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
    
    # Load sample data button
    if st.sidebar.button("📂 Load Sample Data"):
        try:
            import os
            # Try multiple possible locations for sample data
            possible_paths = [
                'sample_data/merovingian_cemetery.csv',
                './sample_data/merovingian_cemetery.csv',
                os.path.join(os.path.dirname(__file__), 'sample_data/merovingian_cemetery.csv'),
                '/home/claude/combitab/sample_data/merovingian_cemetery.csv'
            ]
            
            df = None
            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path, index_col=0)
                    break
            
            if df is None:
                # Create sample data inline if file not found
                sample_data = {
                    'Context': ['Grave_001', 'Grave_002', 'Grave_003', 'Grave_004', 'Grave_005',
                               'Grave_006', 'Grave_007', 'Grave_008', 'Grave_009', 'Grave_010'],
                    'Fibula_bow': [1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    'Fibula_disc': [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                    'Belt_buckle_A': [1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    'Belt_buckle_B': [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
                    'Scramasax': [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    'Spearhead': [1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    'Beads_glass': [1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
                    'Pot_biconical': [1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    'Pot_globular': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    'Comb_double': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                    'Coin_Byzantine': [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                    'Brooch_cloisonne': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
                }
                df = pd.DataFrame(sample_data)
                df = df.set_index('Context')
            
            st.session_state.project.matrix = df
            st.session_state.project.data_type = detect_data_type(df)
            st.session_state.project.name = "Merovingian Cemetery Sample"
            initialize_metadata(st.session_state.project)
            st.session_state.row_order = list(df.index)
            st.session_state.col_order = list(df.columns)
            # Clear any previous file loaded markers
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith('loaded_file_')]
            for k in keys_to_remove:
                del st.session_state[k]
            st.sidebar.success("Sample data loaded!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    st.sidebar.divider()
    
    # Project Import/Export
    st.sidebar.markdown("### 💾 Project File")
    
    # Export button - direct download
    if st.session_state.project.matrix is not None:
        json_str = export_project(st.session_state.project)
        st.sidebar.download_button(
            "📥 Export Project (.json)",
            json_str,
            file_name=f"{st.session_state.project.name.replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.sidebar.button("📥 Export Project (.json)", disabled=True, use_container_width=True)
        st.sidebar.caption("Load data first to enable export")
    
    # Import
    project_file = st.sidebar.file_uploader("Import Project", type=['json'], key='project_import')
    if project_file is not None:
        # Only process if this is a new file
        import_key = f"imported_project_{project_file.name}"
        if import_key not in st.session_state:
            try:
                json_str = project_file.read().decode('utf-8')
                project, row_order, col_order = import_project(json_str)
                st.session_state.project = project
                st.session_state.row_order = row_order if row_order else list(project.matrix.index)
                st.session_state.col_order = col_order if col_order else list(project.matrix.columns)
                # Clear undo/redo stacks on new import
                st.session_state.undo_stack = []
                st.session_state.redo_stack = []
                st.session_state[import_key] = True  # Mark as imported
                st.sidebar.success("Project imported!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Import error: {e}")


def render_main_content():
    """Render main content area"""
    st.markdown('<p class="main-header">CombiTab</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Archaeological Seriation & Combination Table Tool</p>', unsafe_allow_html=True)
    
    if st.session_state.project.matrix is None:
        st.info("👈 Upload a data file or load the sample data to get started.")
        
        st.markdown("""
        ### Getting Started
        
        **CombiTab** helps you create and edit seriation diagrams (combination tables) for archaeological analysis.
        
        **Expected data format:**
        - CSV or Excel file
        - First column: Context names (graves, layers, etc.)
        - Other columns: Artifact types
        - Values: 0/1 for presence/absence, or counts for frequency data
        
        **Features:**
        - Manual row/column reordering
        - Automatic sorting (centroid method)
        - Material group color coding
        - Cell annotations (certainty, fragmentation)
        - Export to PNG, SVG, PDF
        """)
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Matrix View", 
        "📈 Analysis",
        "🔬 Cell Annotations",
        "🏷️ Column Labels", 
        "📝 Row Labels",
        "📤 Export"
    ])
    
    with tab1:
        render_matrix_tab()
    
    with tab2:
        render_analysis_tab()
    
    with tab3:
        render_cell_annotations_tab()
    
    with tab4:
        render_column_metadata_tab()
    
    with tab5:
        render_row_metadata_tab()
    
    with tab6:
        render_export_tab()


def render_matrix_tab():
    """Render the main matrix view and editing interface"""
    project = st.session_state.project
    matrix = project.matrix
    
    # Control bar
    st.markdown("### Sorting Controls")
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
    
    with col1:
        sort_method = st.selectbox(
            "Sort method:",
            ["Centroid", "Correspondence Analysis", "Iterative (Centroid)", "Iterative (CA)"],
            key="sort_method",
            help="Centroid: Simple weighted average. CA: Statistical ordination. Iterative: Repeated sorting until stable."
        )
    
    with col2:
        sort_target = st.selectbox(
            "Sort target:",
            ["Both rows & columns", "Rows only", "Columns only"],
            key="sort_target"
        )
    
    with col3:
        if st.button("🔄 Apply Sorting", use_container_width=True, type="primary"):
            # Save state for undo
            save_state_for_undo()
            
            sort_rows = sort_target != "Columns only"
            sort_cols = sort_target != "Rows only"
            
            # Use current order as starting point
            current_matrix = matrix.loc[st.session_state.row_order, st.session_state.col_order]
            
            if sort_method == "Centroid":
                new_rows, new_cols = sort_by_centroid(
                    current_matrix, project.row_metadata, project.column_metadata,
                    sort_rows=sort_rows, sort_cols=sort_cols
                )
                st.session_state.row_order = list(new_rows)
                st.session_state.col_order = list(new_cols)
                
            elif sort_method == "Correspondence Analysis":
                new_rows, new_cols, ca_info = sort_by_ca(
                    current_matrix, project.row_metadata, project.column_metadata,
                    sort_rows=sort_rows, sort_cols=sort_cols
                )
                st.session_state.row_order = list(new_rows)
                st.session_state.col_order = list(new_cols)
                st.session_state.ca_info = ca_info
                
            elif "Iterative" in sort_method:
                method = 'centroid' if 'Centroid' in sort_method else 'ca'
                new_rows, new_cols, history = iterative_seriation(
                    current_matrix, project.row_metadata, project.column_metadata,
                    max_iterations=10, method=method
                )
                st.session_state.row_order = list(new_rows)
                st.session_state.col_order = list(new_cols)
                st.session_state.quality_history = history
                
            st.rerun()
    
    with col4:
        # Undo button
        undo_disabled = len(st.session_state.undo_stack) == 0
        if st.button("↶ Undo", use_container_width=True, disabled=undo_disabled, 
                     help=f"Undo last change ({len(st.session_state.undo_stack)} steps available)"):
            if undo():
                st.rerun()
    
    with col5:
        # Redo button
        redo_disabled = len(st.session_state.redo_stack) == 0
        if st.button("↷ Redo", use_container_width=True, disabled=redo_disabled,
                     help=f"Redo ({len(st.session_state.redo_stack)} steps available)"):
            if redo():
                st.rerun()
    
    # Second row: Reset button
    reset_col1, reset_col2, reset_col3 = st.columns([2, 2, 4])
    with reset_col1:
        if st.button("↩️ Reset Order", use_container_width=True):
            save_state_for_undo()
            st.session_state.row_order = list(matrix.index)
            st.session_state.col_order = list(matrix.columns)
            st.session_state.ca_info = None
            st.session_state.quality_history = []
            st.rerun()
    
    with reset_col2:
        # Show undo/redo status
        undo_count = len(st.session_state.undo_stack)
        redo_count = len(st.session_state.redo_stack)
        if undo_count > 0 or redo_count > 0:
            st.caption(f"History: {undo_count} undo, {redo_count} redo")
    
    # Matrix size info and filtering
    n_total_rows = len(st.session_state.row_order)
    n_total_cols = len(st.session_state.col_order)
    size_category = get_matrix_size_category(n_total_rows, n_total_cols)
    recommended = get_recommended_settings(size_category)
    
    # Filter & Focus Section
    with st.expander("🔍 Filter & Focus", expanded=False):
        st.markdown(f"**Matrix size:** {n_total_rows} contexts × {n_total_cols} types ({n_total_rows * n_total_cols} cells)")
        
        if recommended["warning"]:
            st.warning(recommended["warning"])
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            st.markdown("**Filter by Material Group:**")
            available_materials = list(project.material_groups.keys())
            selected_materials = st.multiselect(
                "Show only these materials:",
                available_materials,
                default=st.session_state.filter_materials if st.session_state.filter_materials else [],
                key="material_filter_select",
                placeholder="All materials"
            )
            st.session_state.filter_materials = selected_materials
            
            st.markdown("**Hide Empty:**")
            hide_empty_col1, hide_empty_col2 = st.columns(2)
            with hide_empty_col1:
                hide_empty_rows = st.checkbox("Empty rows", 
                                               value=st.session_state.filter_hide_empty_rows,
                                               key="hide_empty_rows_cb")
                st.session_state.filter_hide_empty_rows = hide_empty_rows
            with hide_empty_col2:
                hide_empty_cols = st.checkbox("Empty columns",
                                               value=st.session_state.filter_hide_empty_cols,
                                               key="hide_empty_cols_cb")
                st.session_state.filter_hide_empty_cols = hide_empty_cols
        
        with filter_col2:
            st.markdown("**Focus Mode (Row Range):**")
            focus_rows = st.checkbox("Enable row focus", value=st.session_state.filter_row_range is not None,
                                     key="focus_rows_cb")
            if focus_rows:
                row_range = st.slider(
                    "Select row range:",
                    0, n_total_rows - 1,
                    value=st.session_state.filter_row_range if st.session_state.filter_row_range else (0, min(20, n_total_rows - 1)),
                    key="row_range_slider"
                )
                st.session_state.filter_row_range = row_range
            else:
                st.session_state.filter_row_range = None
            
            st.markdown("**Focus Mode (Column Range):**")
            focus_cols = st.checkbox("Enable column focus", value=st.session_state.filter_col_range is not None,
                                     key="focus_cols_cb")
            if focus_cols:
                col_range = st.slider(
                    "Select column range:",
                    0, n_total_cols - 1,
                    value=st.session_state.filter_col_range if st.session_state.filter_col_range else (0, min(25, n_total_cols - 1)),
                    key="col_range_slider"
                )
                st.session_state.filter_col_range = col_range
            else:
                st.session_state.filter_col_range = None
        
        # Reset filters button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state.filter_materials = []
            st.session_state.filter_row_range = None
            st.session_state.filter_col_range = None
            st.session_state.filter_hide_empty_rows = False
            st.session_state.filter_hide_empty_cols = False
            st.rerun()
    
    # Apply filters to get visible rows/cols
    visible_rows, visible_cols = apply_matrix_filters(
        st.session_state.row_order, 
        st.session_state.col_order, 
        matrix, 
        project
    )
    
    # Show filter status
    n_visible_rows = len(visible_rows)
    n_visible_cols = len(visible_cols)
    if n_visible_rows < n_total_rows or n_visible_cols < n_total_cols:
        st.info(f"📊 Showing {n_visible_rows}/{n_total_rows} contexts × {n_visible_cols}/{n_total_cols} types (filtered)")
    
    # Quality metrics display (on full matrix)
    quality = calculate_seriation_quality(matrix, st.session_state.row_order, st.session_state.col_order)
    
    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Quality Score", f"{quality['quality_score']:.2f}", 
                  help="Overall seriation quality (0-1, higher is better)")
    with metric_cols[1]:
        st.metric("Concentration", f"{quality['concentration']:.2f}",
                  help="How close values are to the diagonal")
    with metric_cols[2]:
        st.metric("Continuity", f"{quality['anti_robinson']:.2f}",
                  help="Absence of gaps within type distributions")
    with metric_cols[3]:
        st.metric("Avg. Type Continuity", f"{quality['avg_continuity']:.2f}",
                  help="Average continuity of type occurrences")
    with metric_cols[4]:
        st.metric("Gap Cells", quality['total_gap_cells'],
                  help="Total number of gap cells in type distributions")
    
    st.divider()
    
    # Edit Mode Toggle and Refresh Button
    edit_col1, edit_col2, edit_col3 = st.columns([2, 2, 4])
    with edit_col1:
        edit_mode = st.checkbox(
            "✏️ Edit Mode (hide matrix)",
            value=st.session_state.edit_mode,
            key="edit_mode_cb",
            help="Hide matrix visualization for faster editing of metadata"
        )
        st.session_state.edit_mode = edit_mode
    
    with edit_col2:
        if st.button("🔄 Refresh Matrix", use_container_width=True, 
                     help="Manually refresh the matrix visualization"):
            st.session_state.matrix_needs_refresh = True
            st.session_state.cached_matrix_fig = None
            st.rerun()
    
    with edit_col3:
        if edit_mode:
            st.info("📊 Matrix hidden for faster editing. Uncheck 'Edit Mode' or click 'Refresh Matrix' when done.")
    
    # Matrix visualization
    col_viz, col_controls = st.columns([3, 1])
    
    with col_viz:
        # Visualization options in expander
        with st.expander("⚙️ Visualization Options", expanded=not edit_mode):
            opt_row0 = st.columns([2, 3])
            with opt_row0[0]:
                use_interactive = st.checkbox(
                    "🔍 Interactive View (Zoom/Pan)",
                    value=st.session_state.use_interactive_view,
                    key="interactive_view_cb",
                    help="Enable interactive zoom, pan, and hover features"
                )
                st.session_state.use_interactive_view = use_interactive
            with opt_row0[1]:
                if use_interactive:
                    st.caption("🖱️ **Controls:** Scroll=Zoom | Drag=Pan | Double-click=Reset | Toolbar: 🔍+/- Pan ⌂Home 📷Save")
            
            opt_row1 = st.columns(4)
            with opt_row1[0]:
                viz_style = st.selectbox(
                    "Style:",
                    ["classic", "battleship", "dots"],
                    index=["classic", "battleship", "dots"].index(st.session_state.viz_style),
                    format_func=lambda x: {"classic": "Classic Squares", 
                                          "battleship": "Battleship Bars",
                                          "dots": "Sized Dots"}[x],
                    key="viz_style_select"
                )
                st.session_state.viz_style = viz_style
            with opt_row1[1]:
                cell_size = st.slider("Cell size:", 0.2, 0.8, 
                                      value=st.session_state.cell_size, 
                                      step=0.05,
                                      key="cell_size_slider")
                st.session_state.cell_size = cell_size
            with opt_row1[2]:
                # Only show "Show values" for frequency data
                if project.data_type == "frequency":
                    show_values = st.checkbox("Show values", 
                                              value=st.session_state.show_values,
                                              key="show_values_cb")
                    st.session_state.show_values = show_values
                else:
                    show_values = False
                    st.session_state.show_values = False
                    st.markdown("*Show values:*")
                    st.caption("(only for frequency data)")
            with opt_row1[3]:
                show_colors = st.checkbox("Material colors", 
                                          value=st.session_state.show_colors,
                                          key="show_colors_cb")
                st.session_state.show_colors = show_colors
            
            opt_row2 = st.columns(4)
            with opt_row2[0]:
                show_certainty = st.checkbox("Show certainty", 
                                             value=st.session_state.show_certainty,
                                             key="show_certainty_cb")
                st.session_state.show_certainty = show_certainty
            with opt_row2[1]:
                show_fragmentation = st.checkbox("Show fragmentation", 
                                                  value=st.session_state.show_fragmentation,
                                                  key="show_fragmentation_cb")
                st.session_state.show_fragmentation = show_fragmentation
        
        # Matrix Display (only if not in edit mode)
        if st.session_state.edit_mode:
            st.markdown("---")
            st.markdown("### 📊 Matrix Preview Hidden")
            st.caption("Matrix visualization is hidden to speed up editing. "
                      "Disable 'Edit Mode' above to see the matrix.")
        elif not visible_rows or not visible_cols:
            st.warning("⚠️ No data to display with current filter settings. Please adjust your filters.")
            st.info("Try: Reset filters, select different material groups, or disable 'Hide empty' options.")
        else:
            # Check if we need to regenerate the figure
            current_viz_params = {
                'row_order': tuple(visible_rows),
                'col_order': tuple(visible_cols),
                'viz_style': viz_style,
                'cell_size': cell_size,
                'show_values': show_values,
                'show_colors': show_colors,
                'show_certainty': show_certainty,
                'use_interactive': use_interactive,
                'data_hash': hash(matrix.to_json()),
                'metadata_hash': hash(str(project.column_metadata))
            }
            
            params_changed = current_viz_params != st.session_state.last_viz_params
            needs_refresh = st.session_state.matrix_needs_refresh or params_changed
            
            # Choose between interactive (Plotly) and static (Matplotlib) view
            if st.session_state.use_interactive_view:
                # Use cached figure if available and no refresh needed
                if st.session_state.cached_matrix_fig is not None and not needs_refresh:
                    st.plotly_chart(st.session_state.cached_matrix_fig, 
                                   width="stretch", 
                                   config=st.session_state.cached_matrix_config)
                else:
                    # Generate new figure
                    fig, config = create_interactive_matrix_figure(
                        matrix, project,
                        visible_rows,
                        visible_cols,
                        cell_size=cell_size,
                        show_values=show_values,
                        show_material_colors=show_colors,
                        show_certainty=show_certainty,
                        visualization_style=viz_style
                    )
                    # Cache the figure
                    st.session_state.cached_matrix_fig = fig
                    st.session_state.cached_matrix_config = config
                    st.session_state.last_viz_params = current_viz_params
                    st.session_state.matrix_needs_refresh = False
                    
                    st.plotly_chart(fig, width="stretch", config=config)
            else:
                # Static Matplotlib view (for export/printing) - no caching
                fig = create_matrix_figure(
                    matrix, project,
                    visible_rows,  # Use filtered rows
                    visible_cols,  # Use filtered cols
                    cell_size=cell_size,
                    show_values=show_values,
                    show_material_colors=show_colors,
                    show_certainty=show_certainty,
                    show_fragmentation=show_fragmentation,
                    visualization_style=viz_style
                )
                st.pyplot(fig)
                plt.close(fig)
                st.session_state.matrix_needs_refresh = False
                st.session_state.last_viz_params = current_viz_params
        
        # Quick editing - Type Statistics & Material Assignment
        with st.expander("📈 Type Statistics & Material Assignment", expanded=False):
            st.markdown("**Edit material groups here** - click 'Apply Changes' when done to update the matrix.")
            
            # Hint about Edit Mode
            if not st.session_state.edit_mode:
                st.caption("💡 Tip: Enable 'Edit Mode' above to hide the matrix and speed up editing large datasets.")
            
            # Sorting and filtering controls
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 2])
            
            with filter_col1:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Type", "Material", "Count", "Contexts", "Span", "First", "Last"],
                    key="stats_sort_by"
                )
            
            with filter_col2:
                sort_order = st.radio(
                    "Order:",
                    ["Ascending", "Descending"],
                    horizontal=True,
                    key="stats_sort_order"
                )
            
            with filter_col3:
                filter_material = st.selectbox(
                    "Filter Material:",
                    ["All"] + list(project.material_groups.keys()),
                    key="stats_filter_material"
                )
            
            with filter_col4:
                filter_text = st.text_input(
                    "Search Type:",
                    "",
                    placeholder="Type name...",
                    key="stats_filter_text"
                )
            
            # Create editable dataframe with statistics AND material assignment
            stats_data = []
            for col in st.session_state.col_order:
                col_data = matrix[col]
                presence_rows = col_data[col_data > 0].index.tolist()
                
                if presence_rows:
                    # Find positions in current order
                    positions = [st.session_state.row_order.index(r) for r in presence_rows 
                                if r in st.session_state.row_order]
                    if positions:
                        first_idx = min(positions)
                        last_idx = max(positions)
                        span = last_idx - first_idx + 1
                    else:
                        first_idx = last_idx = span = 0
                else:
                    first_idx = last_idx = span = 0
                
                col_meta = project.column_metadata.get(col, {})
                
                stats_data.append({
                    'Type': col,
                    'Material': col_meta.get('material_group', 'Unassigned'),
                    'Index Type': col_meta.get('is_index_type', False),
                    'Count': int(col_data.sum()),
                    'Contexts': int((col_data > 0).sum()),
                    'First': first_idx + 1 if presence_rows else 0,
                    'Last': last_idx + 1 if presence_rows else 0,
                    'Span': span
                })
            
            stats_df = pd.DataFrame(stats_data)
            
            # Apply filtering
            if filter_material != "All":
                stats_df = stats_df[stats_df['Material'] == filter_material]
            
            if filter_text:
                stats_df = stats_df[stats_df['Type'].str.contains(filter_text, case=False, na=False)]
            
            # Apply sorting
            ascending = sort_order == "Ascending"
            if sort_by in stats_df.columns:
                stats_df = stats_df.sort_values(by=sort_by, ascending=ascending)
            
            # Show count of filtered results
            total_types = len(st.session_state.col_order)
            shown_types = len(stats_df)
            if shown_types < total_types:
                st.info(f"Showing {shown_types} of {total_types} types")
            
            # Editable dataframe - store in session state for deferred apply
            edited_stats = st.data_editor(
                stats_df,
                column_config={
                    'Type': st.column_config.TextColumn('Type', disabled=True, width="medium"),
                    'Material': st.column_config.SelectboxColumn(
                        'Material',
                        options=list(project.material_groups.keys()),
                        width="small"
                    ),
                    'Index Type': st.column_config.CheckboxColumn('Index', width="small"),
                    'Count': st.column_config.NumberColumn('Count', disabled=True, width="small"),
                    'Contexts': st.column_config.NumberColumn('Ctx', disabled=True, width="small"),
                    'First': st.column_config.NumberColumn('First', disabled=True, width="small"),
                    'Last': st.column_config.NumberColumn('Last', disabled=True, width="small"),
                    'Span': st.column_config.NumberColumn('Span', disabled=True, width="small"),
                },
                hide_index=True,
                width="stretch",
                key="type_stats_editor"
            )
            
            # Store edited data temporarily (don't apply immediately)
            st.session_state.pending_metadata_changes = edited_stats.to_dict('records')
            
            # Apply Changes Button
            st.markdown("---")
            apply_col1, apply_col2, apply_col3 = st.columns([2, 2, 2])
            
            with apply_col1:
                if st.button("✅ Apply Changes", key="apply_metadata_changes", 
                            use_container_width=True, type="primary"):
                    # Apply all pending changes at once
                    changes_count = 0
                    for row in st.session_state.pending_metadata_changes:
                        col_name = row['Type']
                        if col_name not in project.column_metadata:
                            project.column_metadata[col_name] = {'name': col_name}
                        
                        old_material = project.column_metadata[col_name].get('material_group', 'Unassigned')
                        old_index = project.column_metadata[col_name].get('is_index_type', False)
                        
                        if old_material != row['Material'] or old_index != row['Index Type']:
                            changes_count += 1
                        
                        project.column_metadata[col_name]['material_group'] = row['Material']
                        project.column_metadata[col_name]['color'] = project.material_groups.get(row['Material'], '#808080')
                        project.column_metadata[col_name]['is_index_type'] = row['Index Type']
                    
                    # Mark matrix for refresh
                    st.session_state.matrix_needs_refresh = True
                    st.session_state.cached_matrix_fig = None
                    
                    if changes_count > 0:
                        st.success(f"✅ Applied {changes_count} changes!")
                    else:
                        st.info("No changes detected.")
                    
                    # Only rerun if not in edit mode (to refresh matrix)
                    if not st.session_state.edit_mode:
                        st.rerun()
            
            with apply_col2:
                if st.button("🔄 Refresh Table", key="refresh_stats_table", use_container_width=True):
                    st.rerun()
            
            with apply_col3:
                # Show pending changes indicator
                if st.session_state.pending_metadata_changes:
                    pending_count = len(st.session_state.pending_metadata_changes)
                    st.caption(f"📝 {pending_count} types in table")
            
            # Bulk assignment section
            st.markdown("---")
            st.markdown("**Bulk Material Assignment:**")
            bulk_col1, bulk_col2, bulk_col3 = st.columns([2, 2, 1])
            
            with bulk_col1:
                bulk_filter_text = st.text_input(
                    "Types containing:",
                    "",
                    placeholder="e.g. 'Fibula' or 'Pot'",
                    key="bulk_filter_text"
                )
            
            with bulk_col2:
                bulk_material = st.selectbox(
                    "Assign to:",
                    list(project.material_groups.keys()),
                    key="bulk_material"
                )
            
            with bulk_col3:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer
                if st.button("Apply Bulk", key="bulk_apply", use_container_width=True):
                    if bulk_filter_text:
                        count = 0
                        for col in st.session_state.col_order:
                            if bulk_filter_text.lower() in col.lower():
                                if col not in project.column_metadata:
                                    project.column_metadata[col] = {'name': col}
                                project.column_metadata[col]['material_group'] = bulk_material
                                project.column_metadata[col]['color'] = project.material_groups.get(bulk_material, '#808080')
                                count += 1
                        if count > 0:
                            st.success(f"Assigned {count} types to '{bulk_material}'")
                            st.session_state.matrix_needs_refresh = True
                            st.session_state.cached_matrix_fig = None
                            if not st.session_state.edit_mode:
                                st.rerun()
                        else:
                            st.warning("No matching types found")
            
            # Quick material group legend
            st.markdown("---")
            st.markdown("**Material Group Colors:**")
            legend_cols = st.columns(len(project.material_groups))
            for i, (group, color) in enumerate(project.material_groups.items()):
                with legend_cols[i % len(legend_cols)]:
                    st.markdown(f"<span style='color:{color}; font-size:20px;'>■</span> {group}", 
                               unsafe_allow_html=True)
        
        # Cell Annotations Quick Edit
        with st.expander("🔬 Cell Annotations (Quick Edit)", expanded=False):
            st.markdown("**Annotate certainty and fragmentation for finds** - select a context to edit its finds.")
            
            # Context selector
            annot_col1, annot_col2 = st.columns([2, 2])
            
            with annot_col1:
                selected_context = st.selectbox(
                    "Select Context:",
                    st.session_state.row_order,
                    key="quick_annot_context"
                )
            
            with annot_col2:
                # Show summary for selected context
                context_finds = matrix.loc[selected_context]
                n_finds = int((context_finds > 0).sum())
                # Count cells that have been explicitly annotated (user has reviewed them)
                n_annotated = sum(1 for col in st.session_state.col_order 
                                  if context_finds[col] > 0 and f"{selected_context}_{col}" in project.cell_annotations)
                st.metric(f"Finds in {selected_context}", f"{n_annotated}/{n_finds} reviewed")
            
            # Build table of finds for this context
            finds_data = []
            for col in st.session_state.col_order:
                value = context_finds[col]
                if value > 0:  # Only show present finds
                    cell_key = f"{selected_context}_{col}"
                    annot = project.cell_annotations.get(cell_key, {})
                    col_meta = project.column_metadata.get(col, {})
                    
                    # Check if this cell has been explicitly annotated
                    is_reviewed = cell_key in project.cell_annotations
                    
                    finds_data.append({
                        'Type': col,
                        'Material': col_meta.get('material_group', 'Unassigned'),
                        'Value': int(value),
                        'Certainty': annot.get('certainty', 'certain'),
                        'Fragmentation': annot.get('fragmentation', 'unknown'),
                        'Inventory': annot.get('inventory_numbers', ''),
                        'Notes': annot.get('notes', ''),
                        'Reviewed': is_reviewed
                    })
            
            if finds_data:
                finds_df = pd.DataFrame(finds_data)
                
                # Editable dataframe for annotations
                edited_finds = st.data_editor(
                    finds_df,
                    column_config={
                        'Type': st.column_config.TextColumn('Type', disabled=True, width="medium"),
                        'Material': st.column_config.TextColumn('Material', disabled=True, width="small"),
                        'Value': st.column_config.NumberColumn('Val', disabled=True, width="small"),
                        'Certainty': st.column_config.SelectboxColumn(
                            'Certainty',
                            options=['certain', 'uncertain', 'questionable'],
                            width="small"
                        ),
                        'Fragmentation': st.column_config.SelectboxColumn(
                            'Fragmentation',
                            options=['complete', 'fragmentary', 'unknown'],
                            width="small"
                        ),
                        'Inventory': st.column_config.TextColumn('Inventory #', width="medium"),
                        'Notes': st.column_config.TextColumn('Notes', width="medium"),
                        'Reviewed': st.column_config.CheckboxColumn('✓', disabled=True, width="small"),
                    },
                    hide_index=True,
                    width="stretch",
                    key="quick_annot_editor"
                )
                
                # Compare edited data with original to detect actual changes
                for idx, row in edited_finds.iterrows():
                    cell_key = f"{selected_context}_{row['Type']}"
                    original = finds_data[idx]  # Original data from before editing
                    
                    # Check if anything actually changed
                    changed = (
                        row['Certainty'] != original['Certainty'] or
                        row['Fragmentation'] != original['Fragmentation'] or
                        (row['Inventory'] if pd.notna(row['Inventory']) else '') != original['Inventory'] or
                        (row['Notes'] if pd.notna(row['Notes']) else '') != original['Notes']
                    )
                    
                    # Only save if changed OR if user explicitly wants to review
                    if changed:
                        project.cell_annotations[cell_key] = {
                            'certainty': row['Certainty'],
                            'fragmentation': row['Fragmentation'],
                            'inventory_numbers': row['Inventory'] if pd.notna(row['Inventory']) else '',
                            'notes': row['Notes'] if pd.notna(row['Notes']) else ''
                        }
                
                # Bulk annotation for this context
                st.markdown("---")
                st.markdown(f"**Bulk Annotation for {selected_context}:**")
                bulk_annot_col1, bulk_annot_col2, bulk_annot_col3 = st.columns([2, 2, 1])
                
                with bulk_annot_col1:
                    bulk_certainty = st.selectbox(
                        "Set all Certainty to:",
                        ['-- no change --', 'certain', 'uncertain', 'questionable'],
                        key="bulk_certainty_context"
                    )
                
                with bulk_annot_col2:
                    bulk_frag = st.selectbox(
                        "Set all Fragmentation to:",
                        ['-- no change --', 'complete', 'fragmentary', 'unknown'],
                        key="bulk_frag_context"
                    )
                
                with bulk_annot_col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Apply to all", key="bulk_annot_apply", use_container_width=True):
                        count = 0
                        for col in st.session_state.col_order:
                            if context_finds[col] > 0:
                                cell_key = f"{selected_context}_{col}"
                                if cell_key not in project.cell_annotations:
                                    project.cell_annotations[cell_key] = {'certainty': 'certain', 'fragmentation': 'unknown'}
                                if bulk_certainty != '-- no change --':
                                    project.cell_annotations[cell_key]['certainty'] = bulk_certainty
                                if bulk_frag != '-- no change --':
                                    project.cell_annotations[cell_key]['fragmentation'] = bulk_frag
                                count += 1
                        st.success(f"Updated {count} finds in {selected_context}")
                        st.rerun()
                
                # Button to mark all as reviewed (with default values)
                if n_annotated < n_finds:
                    if st.button(f"✓ Mark all {n_finds} finds as reviewed (with current values)", 
                                 key="mark_all_reviewed", use_container_width=True):
                        for col in st.session_state.col_order:
                            if context_finds[col] > 0:
                                cell_key = f"{selected_context}_{col}"
                                if cell_key not in project.cell_annotations:
                                    project.cell_annotations[cell_key] = {
                                        'certainty': 'certain',
                                        'fragmentation': 'unknown',
                                        'inventory_numbers': '',
                                        'notes': ''
                                    }
                        st.success(f"All {n_finds} finds marked as reviewed!")
                        st.rerun()
            else:
                st.info(f"No finds in {selected_context}")
            
            # Global bulk annotation section
            st.markdown("---")
            st.markdown("**Global Bulk Annotation (all contexts):**")
            
            global_col1, global_col2, global_col3, global_col4 = st.columns([2, 2, 2, 1])
            
            with global_col1:
                global_type_filter = st.text_input(
                    "Types containing:",
                    "",
                    placeholder="e.g. 'Fibula'",
                    key="global_annot_type_filter"
                )
            
            with global_col2:
                global_certainty = st.selectbox(
                    "Set Certainty:",
                    ['-- no change --', 'certain', 'uncertain', 'questionable'],
                    key="global_certainty"
                )
            
            with global_col3:
                global_frag = st.selectbox(
                    "Set Fragmentation:",
                    ['-- no change --', 'complete', 'fragmentary', 'unknown'],
                    key="global_frag"
                )
            
            with global_col4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Apply", key="global_annot_apply", use_container_width=True):
                    count = 0
                    for row_idx in st.session_state.row_order:
                        for col in st.session_state.col_order:
                            # Check if type matches filter (or no filter)
                            if global_type_filter and global_type_filter.lower() not in col.lower():
                                continue
                            if matrix.loc[row_idx, col] > 0:
                                cell_key = f"{row_idx}_{col}"
                                if cell_key not in project.cell_annotations:
                                    project.cell_annotations[cell_key] = {}
                                if global_certainty != '-- no change --':
                                    project.cell_annotations[cell_key]['certainty'] = global_certainty
                                if global_frag != '-- no change --':
                                    project.cell_annotations[cell_key]['fragmentation'] = global_frag
                                count += 1
                    if count > 0:
                        st.success(f"Updated {count} cells")
                        st.rerun()
                    else:
                        st.warning("No matching cells found")
    
    with col_controls:
        st.markdown("### Manual Ordering")
        
        # Row reordering
        st.markdown("**Move Row:**")
        selected_row = st.selectbox(
            "Select row",
            st.session_state.row_order,
            key="row_select",
            label_visibility="collapsed"
        )
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            if st.button("⬆️ Up", key="row_up", use_container_width=True):
                current_order = list(st.session_state.row_order)
                idx = current_order.index(selected_row)
                if idx > 0:
                    save_state_for_undo()
                    current_order[idx], current_order[idx-1] = current_order[idx-1], current_order[idx]
                    st.session_state.row_order = current_order
                    st.rerun()
        with r_col2:
            if st.button("⬇️ Down", key="row_down", use_container_width=True):
                current_order = list(st.session_state.row_order)
                idx = current_order.index(selected_row)
                if idx < len(current_order) - 1:
                    save_state_for_undo()
                    current_order[idx], current_order[idx+1] = current_order[idx+1], current_order[idx]
                    st.session_state.row_order = current_order
                    st.rerun()
        
        # Quick row actions
        row_meta = project.row_metadata.get(selected_row, {})
        is_fixed = row_meta.get('is_fixed', False)
        if st.checkbox("🔒 Fix row position", value=is_fixed, key="fix_row"):
            if selected_row not in project.row_metadata:
                project.row_metadata[selected_row] = {'name': selected_row}
            project.row_metadata[selected_row]['is_fixed'] = True
        else:
            if selected_row in project.row_metadata:
                project.row_metadata[selected_row]['is_fixed'] = False
        
        st.divider()
        
        # Column reordering
        st.markdown("**Move Column:**")
        selected_col = st.selectbox(
            "Select column",
            st.session_state.col_order,
            key="col_select",
            label_visibility="collapsed"
        )
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            if st.button("⬅️ Left", key="col_left", use_container_width=True):
                current_order = list(st.session_state.col_order)
                idx = current_order.index(selected_col)
                if idx > 0:
                    save_state_for_undo()
                    current_order[idx], current_order[idx-1] = current_order[idx-1], current_order[idx]
                    st.session_state.col_order = current_order
                    st.rerun()
        with c_col2:
            if st.button("➡️ Right", key="col_right", use_container_width=True):
                current_order = list(st.session_state.col_order)
                idx = current_order.index(selected_col)
                if idx < len(current_order) - 1:
                    save_state_for_undo()
                    current_order[idx], current_order[idx+1] = current_order[idx+1], current_order[idx]
                    st.session_state.col_order = current_order
                    st.rerun()
        
        # Quick column actions
        col_meta = project.column_metadata.get(selected_col, {})
        is_col_fixed = col_meta.get('is_fixed', False)
        if st.checkbox("🔒 Fix column position", value=is_col_fixed, key="fix_col"):
            if selected_col not in project.column_metadata:
                project.column_metadata[selected_col] = {'name': selected_col}
            project.column_metadata[selected_col]['is_fixed'] = True
        else:
            if selected_col in project.column_metadata:
                project.column_metadata[selected_col]['is_fixed'] = False
        
        st.divider()
        
        # Legend
        st.markdown("**Legend:**")
        legend_fig = create_legend_figure(project, show_certainty, show_fragmentation)
        st.pyplot(legend_fig)
        plt.close(legend_fig)


def render_analysis_tab():
    """Render the analysis tab with CA biplot and quality metrics"""
    project = st.session_state.project
    matrix = project.matrix
    
    st.markdown("### Seriation Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Correspondence Analysis")
        
        # Run CA button
        if st.button("🔬 Run Correspondence Analysis", use_container_width=True):
            _, _, ca_info = sort_by_ca(
                matrix, project.row_metadata, project.column_metadata,
                sort_rows=False, sort_cols=False  # Just calculate, don't sort
            )
            st.session_state.ca_info = ca_info
            st.rerun()
        
        # Display CA results
        if st.session_state.ca_info is not None:
            ca_info = st.session_state.ca_info
            
            # Eigenvalues / Inertia
            st.markdown("**Explained Inertia:**")
            
            n_dims = min(5, len(ca_info['eigenvalues']))
            inertia_data = []
            cumulative = 0
            for i in range(n_dims):
                cumulative += ca_info['explained_variance'][i]
                inertia_data.append({
                    'Dimension': i + 1,
                    'Eigenvalue': f"{ca_info['eigenvalues'][i]:.4f}",
                    'Variance %': f"{ca_info['explained_variance'][i]:.1f}%",
                    'Cumulative %': f"{cumulative:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(inertia_data), hide_index=True, width="stretch")
            
            # Scree plot
            fig_scree, ax_scree = plt.subplots(figsize=(6, 3))
            dims = range(1, n_dims + 1)
            ax_scree.bar(dims, ca_info['explained_variance'][:n_dims], color='steelblue', alpha=0.7)
            ax_scree.plot(dims, np.cumsum(ca_info['explained_variance'][:n_dims]), 
                         'ro-', label='Cumulative')
            ax_scree.set_xlabel('Dimension')
            ax_scree.set_ylabel('% Variance')
            ax_scree.set_title('Scree Plot')
            ax_scree.legend()
            plt.tight_layout()
            st.pyplot(fig_scree)
            plt.close(fig_scree)
            
            # CA Biplot
            st.markdown("**CA Biplot:**")
            
            # Biplot display options
            with st.expander("⚙️ Biplot Options", expanded=False):
                bp_col1, bp_col2 = st.columns(2)
                with bp_col1:
                    show_row_labels = st.checkbox("Show context labels", value=True, key="bp_row_labels")
                    show_col_labels = st.checkbox("Show type labels", value=True, key="bp_col_labels")
                with bp_col2:
                    label_size = st.slider("Label size", 5, 12, 8, key="bp_label_size")
                    point_size = st.slider("Point size", 30, 150, 80, key="bp_point_size")
                
                bp_col3, bp_col4 = st.columns(2)
                with bp_col3:
                    arrow_width = st.slider("Arrow width", 0.5, 3.0, 1.5, 0.5, key="bp_arrow_width")
                with bp_col4:
                    fig_size = st.slider("Figure size", 6, 14, 8, key="bp_fig_size")
            
            biplot_fig = create_ca_biplot(
                ca_info, project, 
                figsize=(fig_size, fig_size),
                show_row_labels=show_row_labels,
                show_col_labels=show_col_labels,
                label_font_size=label_size,
                point_size=point_size,
                arrow_width=arrow_width
            )
            st.pyplot(biplot_fig)
            plt.close(biplot_fig)
            
            # Export biplot with callback + fragment pattern
            def generate_biplot():
                biplot_export_fig = create_ca_biplot(
                    ca_info, project, 
                    figsize=(12, 12),
                    show_row_labels=show_row_labels,
                    show_col_labels=show_col_labels,
                    label_font_size=label_size + 2,
                    point_size=point_size * 1.5,
                    arrow_width=arrow_width
                )
                buf = io.BytesIO()
                biplot_export_fig.savefig(buf, format='svg', bbox_inches='tight', facecolor='white')
                plt.close(biplot_export_fig)
                st.session_state.export_biplot_data = buf.getvalue()
            
            st.button("🖼️ Generate High-Res Biplot", on_click=generate_biplot, key="gen_biplot_btn")
            
            if st.session_state.get('export_biplot_data') is not None:
                st.success("✅ Biplot ready!")
                
                @st.fragment
                def biplot_download_fragment():
                    st.download_button(
                        "⬇️ Download Biplot (SVG)",
                        st.session_state.export_biplot_data,
                        file_name=f"{project.name.replace(' ', '_')}_ca_biplot.svg",
                        mime="image/svg+xml"
                    )
                biplot_download_fragment()
        else:
            st.info("Click 'Run Correspondence Analysis' to calculate CA and view the biplot.")
    
    with col2:
        st.markdown("#### Quality Analysis")
        
        # Current quality
        quality = calculate_seriation_quality(
            matrix, st.session_state.row_order, st.session_state.col_order
        )
        
        # Quality gauges
        st.markdown("**Current Seriation Quality:**")
        
        # Create gauge-like display
        metrics = [
            ('Overall Quality', quality['quality_score'], 'Combined score'),
            ('Concentration', quality['concentration'], 'Diagonal concentration'),
            ('Continuity', quality['anti_robinson'], 'No gaps in distributions'),
            ('Type Continuity', quality['avg_continuity'], 'Average type span fill')
        ]
        
        for name, value, desc in metrics:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                # Progress bar as gauge
                st.progress(value, text=f"{name}: {value:.2f}")
            with col_b:
                if value >= 0.8:
                    st.success("Good")
                elif value >= 0.6:
                    st.warning("Fair")
                else:
                    st.error("Poor")
        
        # Gap analysis
        st.markdown("---")
        st.markdown("**Gap Analysis:**")
        
        if quality['gaps']:
            st.warning(f"Found {quality['n_gaps']} types with gaps ({quality['total_gap_cells']} total gap cells)")
            
            with st.expander("View gap details"):
                gap_df = pd.DataFrame(quality['gaps'])
                gap_df = gap_df.sort_values('gaps', ascending=False)
                st.dataframe(gap_df, hide_index=True, width="stretch")
        else:
            st.success("No gaps detected - perfect seriation!")
        
        # Iteration history
        if st.session_state.quality_history:
            st.markdown("---")
            st.markdown("**Sorting History:**")
            
            history_df = pd.DataFrame(st.session_state.quality_history)
            
            fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
            ax_hist.plot(history_df['iteration'], history_df['quality_score'], 
                        'b-o', label='Quality Score')
            ax_hist.plot(history_df['iteration'], history_df['concentration'],
                        'g--', alpha=0.7, label='Concentration')
            ax_hist.plot(history_df['iteration'], history_df['anti_robinson'],
                        'r--', alpha=0.7, label='Continuity')
            ax_hist.set_xlabel('Iteration')
            ax_hist.set_ylabel('Score')
            ax_hist.set_title('Convergence History')
            ax_hist.legend(fontsize=8)
            ax_hist.set_ylim(0, 1.05)
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        
        # Type span visualization
        st.markdown("---")
        st.markdown("**Type Span Overview:**")
        
        if st.checkbox("Show type spans"):
            spans = quality['spans']
            if spans:
                fig_spans, ax_spans = plt.subplots(figsize=(8, max(4, len(spans) * 0.2)))
                
                for i, span_info in enumerate(spans):
                    col_meta = project.column_metadata.get(span_info['column'], {})
                    material = col_meta.get('material_group', 'Unassigned')
                    color = project.material_groups.get(material, '#808080')
                    
                    # Draw span bar
                    ax_spans.barh(i, span_info['span'], left=span_info['first'],
                                 color=color, alpha=0.7, height=0.8)
                    
                    # Mark actual occurrences
                    ax_spans.plot(span_info['first'], i, 'k|', markersize=10)
                    ax_spans.plot(span_info['last'], i, 'k|', markersize=10)
                
                ax_spans.set_yticks(range(len(spans)))
                ax_spans.set_yticklabels([s['column'][:15] for s in spans], fontsize=7)
                ax_spans.set_xlabel('Context Position')
                ax_spans.set_title('Type Spans (colored by material group)')
                ax_spans.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_spans)
                plt.close(fig_spans)


def render_cell_annotations_tab():
    """Render cell-level annotation interface"""
    project = st.session_state.project
    matrix = project.matrix
    
    st.markdown("### Cell Annotations")
    st.markdown("Add certainty, fragmentation, and notes to individual finds.")
    
    # Two-column layout: mini matrix for selection, form for editing
    col_select, col_edit = st.columns([2, 1])
    
    with col_select:
        st.markdown("**Select a cell to annotate:**")
        
        # Row and column selectors
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            selected_row = st.selectbox(
                "Context (row):",
                st.session_state.row_order,
                key="annot_row"
            )
        with sel_col2:
            selected_col = st.selectbox(
                "Type (column):",
                st.session_state.col_order,
                key="annot_col"
            )
        
        # Show current value
        cell_value = matrix.loc[selected_row, selected_col]
        cell_key = f"{selected_row}_{selected_col}"
        
        if cell_value > 0:
            st.success(f"**Value:** {int(cell_value)} {'(present)' if project.data_type == 'presence_absence' else ''}")
        else:
            st.warning("**Value:** 0 (absent) - No annotation needed for empty cells")
        
        # Show mini heatmap of annotations status
        st.markdown("**Annotation Status Overview:**")
        
        # Create a small status matrix
        status_data = []
        for row in st.session_state.row_order[:15]:  # Limit to first 15 rows
            row_status = []
            for col in st.session_state.col_order[:20]:  # Limit to first 20 cols
                val = matrix.loc[row, col]
                key = f"{row}_{col}"
                if val == 0:
                    row_status.append(0)  # Empty
                elif key in project.cell_annotations:
                    annot = project.cell_annotations[key]
                    if annot.get('certainty', 'certain') != 'certain' or \
                       annot.get('fragmentation', 'unknown') != 'unknown' or \
                       annot.get('notes', ''):
                        row_status.append(2)  # Annotated
                    else:
                        row_status.append(1)  # Present, no annotation
                else:
                    row_status.append(1)  # Present, no annotation
            status_data.append(row_status)
        
        # Display as colored grid
        fig_status, ax_status = plt.subplots(figsize=(8, 4))
        status_colors = ['white', 'lightgray', '#90EE90']  # Empty, present, annotated
        cmap = plt.cm.colors.ListedColormap(status_colors)
        
        im = ax_status.imshow(status_data, cmap=cmap, aspect='auto', vmin=0, vmax=2)
        ax_status.set_xticks(range(min(20, len(st.session_state.col_order))))
        ax_status.set_xticklabels([c[:8] for c in st.session_state.col_order[:20]], 
                                   rotation=90, fontsize=6)
        ax_status.set_yticks(range(min(15, len(st.session_state.row_order))))
        ax_status.set_yticklabels([r[:10] for r in st.session_state.row_order[:15]], fontsize=6)
        ax_status.set_title("Gray = present, Green = annotated", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_status)
        plt.close(fig_status)
    
    with col_edit:
        st.markdown("**Edit Annotation:**")
        
        if cell_value == 0:
            st.info("Select a cell with a find to add annotations.")
        else:
            # Get existing annotation or create default
            current_annot = project.cell_annotations.get(cell_key, {})
            
            # Certainty
            certainty = st.radio(
                "Certainty of identification:",
                ["certain", "uncertain", "questionable"],
                index=["certain", "uncertain", "questionable"].index(
                    current_annot.get('certainty', 'certain')
                ),
                key="edit_certainty",
                help="How confident is the type identification?"
            )
            
            # Fragmentation
            fragmentation = st.radio(
                "Fragmentation:",
                ["complete", "fragmentary", "unknown"],
                index=["complete", "fragmentary", "unknown"].index(
                    current_annot.get('fragmentation', 'unknown')
                ),
                key="edit_frag",
                help="Is the artifact complete or fragmentary?"
            )
            
            # Count range (for frequency data)
            if project.data_type == "frequency":
                st.markdown("**Count range (if uncertain):**")
                count_cols = st.columns(2)
                with count_cols[0]:
                    count_min = st.number_input(
                        "Min:",
                        min_value=0,
                        value=current_annot.get('count_min') or int(cell_value),
                        key="edit_count_min"
                    )
                with count_cols[1]:
                    count_max = st.number_input(
                        "Max:",
                        min_value=0,
                        value=current_annot.get('count_max') or int(cell_value),
                        key="edit_count_max"
                    )
            else:
                count_min = count_max = None
            
            # Inventory numbers
            inventory = st.text_input(
                "Inventory number(s):",
                value=current_annot.get('inventory_numbers', ''),
                key="edit_inv",
                help="Comma-separated inventory numbers"
            )
            
            # Notes
            notes = st.text_area(
                "Notes:",
                value=current_annot.get('notes', ''),
                key="edit_notes",
                height=80
            )
            
            # Save button
            if st.button("💾 Save Annotation", use_container_width=True, type="primary"):
                project.cell_annotations[cell_key] = {
                    'certainty': certainty,
                    'fragmentation': fragmentation,
                    'count_min': count_min,
                    'count_max': count_max,
                    'inventory_numbers': inventory,
                    'notes': notes
                }
                st.success("Annotation saved!")
                st.rerun()
            
            # Clear button
            if cell_key in project.cell_annotations:
                if st.button("🗑️ Clear Annotation", use_container_width=True):
                    del project.cell_annotations[cell_key]
                    st.success("Annotation cleared!")
                    st.rerun()
    
    # Bulk annotation section
    st.divider()
    st.markdown("### Bulk Operations")
    
    bulk_col1, bulk_col2 = st.columns(2)
    
    with bulk_col1:
        st.markdown("**Set certainty for all finds of a type:**")
        bulk_type = st.selectbox(
            "Select type:",
            st.session_state.col_order,
            key="bulk_type"
        )
        bulk_certainty = st.selectbox(
            "Set certainty to:",
            ["certain", "uncertain", "questionable"],
            key="bulk_certainty"
        )
        if st.button("Apply to all contexts", key="bulk_apply_type"):
            count = 0
            for row in st.session_state.row_order:
                if matrix.loc[row, bulk_type] > 0:
                    cell_key = f"{row}_{bulk_type}"
                    if cell_key not in project.cell_annotations:
                        project.cell_annotations[cell_key] = {}
                    project.cell_annotations[cell_key]['certainty'] = bulk_certainty
                    count += 1
            st.success(f"Updated {count} cells!")
            st.rerun()
    
    with bulk_col2:
        st.markdown("**Set fragmentation for all finds in a context:**")
        bulk_context = st.selectbox(
            "Select context:",
            st.session_state.row_order,
            key="bulk_context"
        )
        bulk_frag = st.selectbox(
            "Set fragmentation to:",
            ["complete", "fragmentary", "unknown"],
            key="bulk_frag"
        )
        if st.button("Apply to all types", key="bulk_apply_context"):
            count = 0
            for col in st.session_state.col_order:
                if matrix.loc[bulk_context, col] > 0:
                    cell_key = f"{bulk_context}_{col}"
                    if cell_key not in project.cell_annotations:
                        project.cell_annotations[cell_key] = {}
                    project.cell_annotations[cell_key]['fragmentation'] = bulk_frag
                    count += 1
            st.success(f"Updated {count} cells!")
            st.rerun()
    
    # Export annotations summary
    st.divider()
    if st.checkbox("Show all annotations"):
        if project.cell_annotations:
            annot_list = []
            for cell_key, annot in project.cell_annotations.items():
                row, col = cell_key.rsplit('_', 1)
                # Handle column names with underscores
                for c in st.session_state.col_order:
                    if cell_key.startswith(f"{row}_") or cell_key.endswith(f"_{c}"):
                        parts = cell_key.split('_')
                        # Find the split point
                        for i in range(1, len(parts)):
                            potential_row = '_'.join(parts[:i])
                            potential_col = '_'.join(parts[i:])
                            if potential_row in st.session_state.row_order and \
                               potential_col in st.session_state.col_order:
                                row, col = potential_row, potential_col
                                break
                        break
                
                annot_list.append({
                    'Context': row,
                    'Type': col,
                    'Value': matrix.loc[row, col] if row in matrix.index and col in matrix.columns else '-',
                    'Certainty': annot.get('certainty', '-'),
                    'Fragmentation': annot.get('fragmentation', '-'),
                    'Inventory': annot.get('inventory_numbers', ''),
                    'Notes': annot.get('notes', '')[:30] + '...' if len(annot.get('notes', '')) > 30 else annot.get('notes', '')
                })
            
            annot_df = pd.DataFrame(annot_list)
            st.dataframe(annot_df, hide_index=True, width="stretch")
        else:
            st.info("No annotations yet.")


def render_column_metadata_tab():
    """Render column metadata editing interface"""
    project = st.session_state.project
    
    st.markdown("### Artifact Type Labels")
    st.markdown("Assign material groups and other attributes to columns.")
    
    # Material group management
    with st.expander("⚙️ Manage Material Groups"):
        new_group = st.text_input("Add new group:")
        new_color = st.color_picker("Color:", "#666666")
        if st.button("Add Group") and new_group:
            project.material_groups[new_group] = new_color
            st.rerun()
        
        st.markdown("**Current groups:**")
        for group, color in project.material_groups.items():
            st.markdown(f"<span style='color:{color}'>●</span> {group}", unsafe_allow_html=True)
    
    # Column metadata editor
    st.divider()
    
    # Create editable dataframe for column metadata
    col_data = []
    for col in st.session_state.col_order:
        meta = project.column_metadata.get(col, {})
        col_data.append({
            'Column': col,
            'Material Group': meta.get('material_group', 'Unassigned'),
            'Index Type': meta.get('is_index_type', False),
            'Fixed': meta.get('is_fixed', False),
            'Notes': meta.get('notes', '')
        })
    
    df_cols = pd.DataFrame(col_data)
    
    edited_df = st.data_editor(
        df_cols,
        column_config={
            'Column': st.column_config.TextColumn('Column', disabled=True),
            'Material Group': st.column_config.SelectboxColumn(
                'Material Group',
                options=list(project.material_groups.keys())
            ),
            'Index Type': st.column_config.CheckboxColumn('Index Type'),
            'Fixed': st.column_config.CheckboxColumn('Fixed (no auto-sort)'),
            'Notes': st.column_config.TextColumn('Notes')
        },
        hide_index=True,
        width="stretch"
    )
    
    # Update metadata from edited dataframe
    for _, row in edited_df.iterrows():
        col_name = row['Column']
        project.column_metadata[col_name] = {
            'name': col_name,
            'material_group': row['Material Group'],
            'color': project.material_groups.get(row['Material Group'], '#808080'),
            'is_index_type': row['Index Type'],
            'is_fixed': row['Fixed'],
            'notes': row['Notes']
        }


def render_row_metadata_tab():
    """Render row metadata editing interface"""
    project = st.session_state.project
    
    st.markdown("### Context Labels")
    st.markdown("Assign context types and other attributes to rows.")
    
    # Context type management
    with st.expander("⚙️ Manage Context Types"):
        new_type = st.text_input("Add new context type:")
        if st.button("Add Type") and new_type:
            project.context_types.append(new_type)
            st.rerun()
        
        st.markdown("**Current types:** " + ", ".join(project.context_types))
    
    st.divider()
    
    # Row metadata editor
    row_data = []
    for row in st.session_state.row_order:
        meta = project.row_metadata.get(row, {})
        row_data.append({
            'Context': row,
            'Type': meta.get('context_type', 'Unassigned'),
            'Area': meta.get('area', ''),
            'Fixed': meta.get('is_fixed', False),
            'Notes': meta.get('notes', '')
        })
    
    df_rows = pd.DataFrame(row_data)
    
    edited_df = st.data_editor(
        df_rows,
        column_config={
            'Context': st.column_config.TextColumn('Context', disabled=True),
            'Type': st.column_config.SelectboxColumn(
                'Type',
                options=project.context_types
            ),
            'Area': st.column_config.TextColumn('Area/Trench'),
            'Fixed': st.column_config.CheckboxColumn('Fixed (no auto-sort)'),
            'Notes': st.column_config.TextColumn('Notes')
        },
        hide_index=True,
        width="stretch"
    )
    
    # Update metadata
    for _, row in edited_df.iterrows():
        row_name = row['Context']
        project.row_metadata[row_name] = {
            'name': row_name,
            'context_type': row['Type'],
            'area': row['Area'],
            'is_fixed': row['Fixed'],
            'notes': row['Notes']
        }


def render_export_tab():
    """Render export options"""
    project = st.session_state.project
    
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Matrix Image")
        
        export_format = st.selectbox(
            "Format:",
            ["PNG", "SVG", "PDF"]
        )
        
        dpi = st.slider("Resolution (DPI):", 72, 600, 300)
        
        viz_style = st.selectbox(
            "Visualization style:",
            ["classic", "battleship", "dots"],
            format_func=lambda x: {"classic": "Classic Squares", 
                                  "battleship": "Battleship Bars",
                                  "dots": "Sized Dots"}[x],
            key="exp_viz_style"
        )
        
        exp_opts = st.columns(2)
        with exp_opts[0]:
            show_vals = st.checkbox("Show values", value=False, key="exp_vals")
            show_cols = st.checkbox("Material colors", value=True, key="exp_cols")
        with exp_opts[1]:
            show_cert = st.checkbox("Show certainty", value=True, key="exp_cert")
            show_frag = st.checkbox("Show fragmentation", value=False, key="exp_frag")
        
        cell_sz = st.slider("Cell size:", 0.2, 1.0, 0.5, 0.1)
        
        format_map = {"PNG": "png", "SVG": "svg", "PDF": "pdf"}
        mime_map = {
            "PNG": "image/png",
            "SVG": "image/svg+xml", 
            "PDF": "application/pdf"
        }
        
        # Generate image on button click and store in session state
        def generate_matrix_image():
            fig = create_matrix_figure(
                project.matrix, project,
                st.session_state.row_order,
                st.session_state.col_order,
                cell_size=cell_sz,
                show_values=show_vals,
                show_material_colors=show_cols,
                show_certainty=show_cert,
                show_fragmentation=show_frag,
                visualization_style=viz_style
            )
            buf = io.BytesIO()
            fig.savefig(buf, format=format_map[export_format], dpi=dpi, 
                       bbox_inches='tight', facecolor='white')
            plt.close(fig)
            st.session_state.export_matrix_data = buf.getvalue()
            st.session_state.export_matrix_format = export_format
            st.session_state.export_matrix_filename = f"{project.name.replace(' ', '_')}_matrix.{format_map[export_format]}"
            st.session_state.export_matrix_mime = mime_map[export_format]
        
        st.button("🖼️ Generate Matrix Image", on_click=generate_matrix_image, use_container_width=True)
        
        # Show download in fragment if data exists
        if st.session_state.get('export_matrix_data') is not None:
            st.success(f"✅ {st.session_state.export_matrix_format} ready!")
            
            @st.fragment
            def matrix_download_fragment():
                st.download_button(
                    f"⬇️ Download {st.session_state.export_matrix_format}",
                    st.session_state.export_matrix_data,
                    file_name=st.session_state.export_matrix_filename,
                    mime=st.session_state.export_matrix_mime,
                    use_container_width=True
                )
            matrix_download_fragment()
        
        # Export legend
        st.markdown("---")
        st.markdown("**Export Legend:**")
        
        def generate_legend():
            legend_fig = create_legend_figure(project, show_cert, show_frag)
            buf = io.BytesIO()
            legend_fig.savefig(buf, format='svg', bbox_inches='tight', facecolor='white')
            plt.close(legend_fig)
            st.session_state.export_legend_data = buf.getvalue()
        
        st.button("🏷️ Generate Legend", on_click=generate_legend, use_container_width=True)
        
        if st.session_state.get('export_legend_data') is not None:
            st.success("✅ Legend ready!")
            
            @st.fragment
            def legend_download_fragment():
                st.download_button(
                    "⬇️ Download Legend (SVG)",
                    st.session_state.export_legend_data,
                    file_name=f"{project.name.replace(' ', '_')}_legend.svg",
                    mime="image/svg+xml",
                    use_container_width=True
                )
            legend_download_fragment()
    
    with col2:
        st.markdown("#### 📄 Data Export")
        
        data_format = st.selectbox(
            "Format:",
            ["CSV (sorted matrix)", "Excel (with metadata)", "Excel (with annotations)"]
        )
        
        def generate_data_file():
            sorted_matrix = project.matrix.loc[
                st.session_state.row_order,
                st.session_state.col_order
            ]
            
            if "CSV" in data_format:
                csv_buf = io.StringIO()
                sorted_matrix.to_csv(csv_buf)
                st.session_state.export_data_content = csv_buf.getvalue()
                st.session_state.export_data_filename = f"{project.name.replace(' ', '_')}_sorted.csv"
                st.session_state.export_data_mime = "text/csv"
                st.session_state.export_data_label = "CSV"
            else:
                xlsx_buf = io.BytesIO()
                with pd.ExcelWriter(xlsx_buf, engine='openpyxl') as writer:
                    sorted_matrix.to_excel(writer, sheet_name='Matrix')
                    
                    col_meta_df = pd.DataFrame([
                        project.column_metadata.get(c, {'name': c})
                        for c in st.session_state.col_order
                    ])
                    col_meta_df.to_excel(writer, sheet_name='Column_Metadata', index=False)
                    
                    row_meta_df = pd.DataFrame([
                        project.row_metadata.get(r, {'name': r})
                        for r in st.session_state.row_order
                    ])
                    row_meta_df.to_excel(writer, sheet_name='Row_Metadata', index=False)
                    
                    if "annotations" in data_format.lower() and project.cell_annotations:
                        annot_list = []
                        for cell_key, annot in project.cell_annotations.items():
                            for r in st.session_state.row_order:
                                for c in st.session_state.col_order:
                                    if cell_key == f"{r}_{c}":
                                        annot_list.append({
                                            'Context': r, 'Type': c,
                                            'Value': sorted_matrix.loc[r, c],
                                            **annot
                                        })
                                        break
                        if annot_list:
                            pd.DataFrame(annot_list).to_excel(writer, sheet_name='Cell_Annotations', index=False)
                    
                    stats_df = create_summary_statistics(
                        project.matrix, project,
                        st.session_state.row_order,
                        st.session_state.col_order
                    )
                    stats_df.to_excel(writer, sheet_name='Type_Statistics', index=False)
                
                st.session_state.export_data_content = xlsx_buf.getvalue()
                st.session_state.export_data_filename = f"{project.name.replace(' ', '_')}_complete.xlsx"
                st.session_state.export_data_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                st.session_state.export_data_label = "Excel"
        
        st.button("📊 Generate Data File", on_click=generate_data_file, use_container_width=True)
        
        if st.session_state.get('export_data_content') is not None:
            st.success(f"✅ {st.session_state.export_data_label} ready!")
            
            @st.fragment
            def data_download_fragment():
                st.download_button(
                    f"⬇️ Download {st.session_state.export_data_label}",
                    st.session_state.export_data_content,
                    file_name=st.session_state.export_data_filename,
                    mime=st.session_state.export_data_mime,
                    use_container_width=True
                )
            data_download_fragment()
        
        st.markdown("---")
        st.markdown("#### 📋 Quick Statistics")
        
        # Show summary stats
        n_cells = len(st.session_state.row_order) * len(st.session_state.col_order)
        n_present = (project.matrix > 0).sum().sum()
        n_annotated = len(project.cell_annotations)
        
        st.metric("Total cells", n_cells)
        st.metric("Cells with finds", int(n_present))
        st.metric("Annotated cells", n_annotated)
        st.metric("Annotation coverage", f"{n_annotated / max(1, n_present) * 100:.1f}%")


# ============================================================
# KEYBOARD SHORTCUTS
# ============================================================

def add_keyboard_shortcuts():
    """Add keyboard shortcuts using JavaScript injection"""
    # JavaScript for keyboard shortcuts
    keyboard_js = """
    <script>
    document.addEventListener('keydown', function(e) {
        // Only trigger if not in input/textarea
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        // Ctrl+Z: Undo
        if (e.ctrlKey && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            // Find and click the Undo button
            const undoBtn = Array.from(document.querySelectorAll('button')).find(
                btn => btn.innerText.includes('Undo')
            );
            if (undoBtn && !undoBtn.disabled) undoBtn.click();
        }
        
        // Ctrl+Y or Ctrl+Shift+Z: Redo
        if ((e.ctrlKey && e.key === 'y') || (e.ctrlKey && e.shiftKey && e.key === 'z')) {
            e.preventDefault();
            const redoBtn = Array.from(document.querySelectorAll('button')).find(
                btn => btn.innerText.includes('Redo')
            );
            if (redoBtn && !redoBtn.disabled) redoBtn.click();
        }
        
        // Ctrl+S: Export (prevent browser save dialog)
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            const exportBtn = Array.from(document.querySelectorAll('button')).find(
                btn => btn.innerText.includes('Export Project')
            );
            if (exportBtn) exportBtn.click();
        }
    });
    </script>
    """
    st.components.v1.html(keyboard_js, height=0)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    init_session_state()
    add_keyboard_shortcuts()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
