#!/usr/bin/env python3
"""
Professional Streamlit Web Application for Hardware-Adaptive RAG System
Features: File-aware chunking (background), recursive chunking, safety detection, evaluation pipeline, performance analytics
"""

import streamlit as st
import time
import json
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import sys
import os
import shutil

# Add the current directory to Python path to import our enhanced RAG system
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our enhanced RAG system components
try:
    # Suppress PyTorch warnings that conflict with Streamlit
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    from main import (
        HardwareDetector, RAGConfig, HardwareAdaptiveRAGSystem,
        DocumentProcessor, HardwareAdaptiveEmbeddingGenerator,
        HardwareAdaptiveVectorStore, OllamaClient, RAGEvaluator,
        SafetyKeywordDetector, RecursiveTextSplitter, FileAwareChunker
    )
except ImportError as e:
    st.error(f"Could not import enhanced RAG system components: {e}")
    st.info("Make sure main.py is in the same directory")
    st.stop()
except Exception as e:
    st.warning(f"Import warning (system should still work): {e}")
    try:
        from main import (
            HardwareDetector, RAGConfig, HardwareAdaptiveRAGSystem,
            DocumentProcessor, HardwareAdaptiveEmbeddingGenerator,
            HardwareAdaptiveVectorStore, OllamaClient, RAGEvaluator,
            SafetyKeywordDetector, RecursiveTextSplitter, FileAwareChunker
        )
    except Exception as e2:
        st.error(f"Could not import enhanced RAG system components: {e2}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Electrical Utility RAG System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling with dark mode support
PROFESSIONAL_CSS = """
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* CSS Custom Properties for theme support */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f7fafc;
        --bg-tertiary: #edf2f7;
        --text-primary: #1a202c;
        --text-secondary: #2d3748;
        --text-tertiary: #4a5568;
        --text-muted: #718096;
        --border-primary: #e2e8f0;
        --border-secondary: #cbd5e0;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --blue-primary: #3182ce;
        --blue-secondary: #2c5282;
        --green-primary: #38a169;
        --green-secondary: #48bb78;
        --orange-primary: #ed8936;
        --red-primary: #e53e3e;
    }

    /* Dark mode CSS custom properties */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1a202c;
            --bg-secondary: #2d3748;
            --bg-tertiary: #4a5568;
            --text-primary: #f7fafc;
            --text-secondary: #e2e8f0;
            --text-tertiary: #cbd5e0;
            --text-muted: #a0aec0;
            --border-primary: #4a5568;
            --border-secondary: #718096;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.3);
            --blue-primary: #4299e1;
            --blue-secondary: #3182ce;
            --green-primary: #48bb78;
            --green-secondary: #38a169;
            --orange-primary: #fbb53c;
            --red-primary: #fc8181;
        }
    }

    /* FULL WIDTH CONTAINER */
    .main > div.block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    .main {
        width: 100% !important;
        max-width: none !important;
    }

    div[data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 !important;
    }

    /* Professional Typography with theme support */
    .main h1, div[data-testid="stMarkdownContainer"] h1 {
        font-size: clamp(1.8rem, 4vw, 2.5rem) !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        width: 100% !important;
    }

    .main h2, div[data-testid="stMarkdownContainer"] h2 {
        font-size: clamp(1.4rem, 3vw, 1.8rem) !important;
        margin-bottom: 0.5rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        width: 100% !important;
    }

    .main h3, div[data-testid="stMarkdownContainer"] h3 {
        font-size: clamp(1.2rem, 2.5vw, 1.5rem) !important;
        margin-bottom: 0.5rem !important;
        font-weight: 500 !important;
        color: var(--text-tertiary) !important;
        width: 100% !important;
    }

    .main h4, div[data-testid="stMarkdownContainer"] h4 {
        font-size: clamp(1rem, 2vw, 1.2rem) !important;
        margin-bottom: 0.4rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        width: 100% !important;
    }

    /* Professional text classes with theme support */
    div[data-testid="stMarkdownContainer"] .small-text,
    .main .small-text {
        font-size: clamp(0.8rem, 1.5vw, 0.9rem) !important;
        line-height: 1.5 !important;
        color: var(--text-tertiary) !important;
        width: 100% !important;
    }

    div[data-testid="stMarkdownContainer"] .tiny-text,
    .main .tiny-text {
        font-size: clamp(0.7rem, 1.2vw, 0.8rem) !important;
        line-height: 1.4 !important;
        color: var(--text-muted) !important;
        width: 100% !important;
    }

    /* Professional cards with theme support */
    div[data-testid="stMarkdownContainer"] .info-card,
    .main .info-card {
        background: var(--bg-primary) !important;
        padding: 1.5rem !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
        border: 1px solid var(--border-primary) !important;
        box-shadow: var(--shadow-sm) !important;
        width: 100% !important;
        box-sizing: border-box !important;
        color: var(--text-secondary) !important;
    }

    div[data-testid="stMarkdownContainer"] .status-card,
    .main .status-card {
        background: var(--bg-primary) !important;
        padding: 1rem !important;
        border-radius: 6px !important;
        border-left: 4px solid var(--blue-primary) !important;
        margin: 0.5rem 0 !important;
        font-size: clamp(0.8rem, 1.5vw, 0.95rem) !important;
        box-shadow: var(--shadow-sm) !important;
        width: 100% !important;
        box-sizing: border-box !important;
        color: var(--text-secondary) !important;
    }

    /* Professional status indicators - consistent across themes */
    .tier-ULTRA { color: var(--green-primary) !important; font-weight: 600 !important; }
    .tier-HIGH { color: var(--green-secondary) !important; font-weight: 600 !important; }
    .tier-MEDIUM { color: var(--orange-primary) !important; font-weight: 600 !important; }
    .tier-LOW { color: var(--red-primary) !important; font-weight: 600 !important; }

    .safety-HIGH { color: var(--red-primary) !important; font-weight: 500 !important; }
    .safety-MEDIUM { color: var(--orange-primary) !important; font-weight: 500 !important; }
    .safety-LOW { color: var(--green-primary) !important; font-weight: 500 !important; }

    /* Professional metric containers with theme support */
    div[data-testid="stMarkdownContainer"] .metric-container,
    .main .metric-container {
        background: var(--bg-primary) !important;
        padding: clamp(0.75rem, 2vw, 1rem) !important;
        border-radius: 8px !important;
        text-align: center !important;
        border: 1px solid var(--border-primary) !important;
        margin: 0.25rem 0 !important;
        box-shadow: var(--shadow-sm) !important;
        width: 100% !important;
        box-sizing: border-box !important;
        min-height: 80px !important;
    }

    div[data-testid="stMarkdownContainer"] .metric-value,
    .main .metric-value {
        font-size: clamp(1.2rem, 3vw, 1.8rem) !important;
        font-weight: 600 !important;
        color: var(--blue-primary) !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }

    div[data-testid="stMarkdownContainer"] .metric-label,
    .main .metric-label {
        font-size: clamp(0.7rem, 1.2vw, 0.85rem) !important;
        color: var(--text-muted) !important;
        margin: 0 !important;
        margin-top: 0.5rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Professional status indicators */
    .status-success { color: var(--green-primary) !important; font-weight: 600 !important; }
    .status-warning { color: var(--orange-primary) !important; font-weight: 600 !important; }
    .status-error { color: var(--red-primary) !important; font-weight: 600 !important; }

    /* Professional buttons with theme support */
    div[data-testid="stButton"] > button,
    .main button {
        border-radius: 6px !important;
        border: 1px solid var(--border-secondary) !important;
        font-size: clamp(0.8rem, 1.5vw, 0.95rem) !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-secondary) !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stButton"] > button:hover,
    .main button:hover {
        background-color: var(--bg-secondary) !important;
        border-color: var(--text-muted) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"],
    .main button[kind="primary"] {
        background-color: var(--blue-primary) !important;
        color: white !important;
        border-color: var(--blue-primary) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: var(--blue-secondary) !important;
        border-color: var(--blue-secondary) !important;
    }

    /* Professional dataframes with theme support */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] table,
    .main .dataframe {
        font-size: clamp(0.75rem, 1.3vw, 0.9rem) !important;
        font-family: 'Inter', sans-serif !important;
        width: 100% !important;
        overflow-x: auto !important;
    }

    div[data-testid="stDataFrame"] th {
        background-color: var(--bg-secondary) !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Professional sidebar with theme support */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-primary) !important;
        min-width: 280px !important;
        max-width: 350px !important;
    }

    section[data-testid="stSidebar"] .main {
        padding-top: 1rem !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-secondary) !important;
        font-size: clamp(0.9rem, 2vw, 1.1rem) !important;
    }

    /* Ensure sidebar text is visible in both themes */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: var(--text-secondary) !important;
    }

    /* Professional alerts with theme support */
    div[data-testid="stAlert"],
    div[data-testid="stSuccess"],
    div[data-testid="stInfo"],
    div[data-testid="stWarning"],
    div[data-testid="stError"] {
        border-radius: 8px !important;
        font-size: clamp(0.8rem, 1.5vw, 0.95rem) !important;
        font-weight: 500 !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Professional input fields with theme support */
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        border-radius: 6px !important;
        border: 1px solid var(--border-secondary) !important;
        font-size: clamp(0.8rem, 1.5vw, 0.95rem) !important;
        width: 100% !important;
        box-sizing: border-box !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-secondary) !important;
    }

    /* Fix expandable sections for dark mode */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-primary) !important;
        border-radius: 6px !important;
        background-color: var(--bg-primary) !important;
    }

    div[data-testid="stExpander"] summary {
        color: var(--text-secondary) !important;
        background-color: var(--bg-secondary) !important;
    }

    /* Fix code blocks for dark mode */
    div[data-testid="stCodeBlock"],
    .main pre,
    code {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-primary) !important;
    }

    /* Fix metric widgets for dark mode */
    div[data-testid="metric-container"] {
        background-color: var(--bg-primary) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-secondary) !important;
    }

    /* Ensure radio buttons are visible in dark mode */
    div[data-testid="stRadio"] label {
        color: var(--text-secondary) !important;
    }

    /* Fix progress bars for dark mode */
    .stProgress > div {
        background-color: var(--border-primary) !important;
    }

    .stProgress > div > div {
        background-color: var(--blue-primary) !important;
    }

    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Responsive design */
    @media (max-width: 1200px) {
        .main > div.block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
    }

    @media (max-width: 768px) {
        .main > div.block-container {
            padding: 1rem 0.5rem !important;
        }
    }

    /* Dark mode specific fixes for Streamlit elements */
    @media (prefers-color-scheme: dark) {
        /* Ensure text in custom divs is visible */
        .small-text, .tiny-text {
            color: var(--text-tertiary) !important;
        }
        
        /* Fix any remaining white text on white background issues */
        div[style*="background-color: #f8fafc"] {
            background-color: var(--bg-secondary) !important;
            color: var(--text-secondary) !important;
        }
        
        /* Ensure all custom styled elements respect dark mode */
        .info-card *, .status-card *, .metric-container * {
            color: inherit !important;
        }
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# Professional layout functions
def display_professional_hardware_info(hardware_detector):
    """Display hardware information with professional styling"""
    hw_info = hardware_detector.hardware_info

    st.markdown("### Hardware Configuration")

    # Hardware tier with professional styling
    tier_class = f"tier-{hw_info.performance_tier}"
    st.markdown(
        f'<div class="status-card">Performance Tier: <span class="{tier_class}">{hw_info.performance_tier}</span></div>',
        unsafe_allow_html=True)

    # Professional metrics grid
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-container"><div class="metric-value">{hw_info.cpu_count}</div><div class="metric-label">CPU Cores</div></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="metric-container"><div class="metric-value">{hw_info.cpu_freq_max:.1f}</div><div class="metric-label">CPU GHz</div></div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            f'<div class="metric-container"><div class="metric-value">{hw_info.total_ram_gb:.1f}</div><div class="metric-label">Total RAM</div></div>',
            unsafe_allow_html=True)
    with col4:
        st.markdown(
            f'<div class="metric-container"><div class="metric-value">{hw_info.available_ram_gb:.1f}</div><div class="metric-label">Available RAM</div></div>',
            unsafe_allow_html=True)

    with col5:
        gpu_display = hw_info.gpu_name[:15] + "..." if len(hw_info.gpu_name) > 15 else hw_info.gpu_name
        st.markdown(
            f'<div class="metric-container"><div class="metric-value" style="font-size:clamp(0.8rem, 2vw, 1rem);">{gpu_display}</div><div class="metric-label">GPU</div></div>',
            unsafe_allow_html=True)
    with col6:
        vram_val = f"{hw_info.gpu_vram_gb:.1f}" if hw_info.has_gpu else "CPU"
        st.markdown(
            f'<div class="metric-container"><div class="metric-value">{vram_val}</div><div class="metric-label">VRAM GB</div></div>',
            unsafe_allow_html=True)
    with col7:
        st.markdown(
            f'<div class="metric-container"><div class="metric-value" style="font-size:clamp(0.8rem, 2vw, 1rem);">{hw_info.storage_type}</div><div class="metric-label">Storage</div></div>',
            unsafe_allow_html=True)
    with col8:
        st.markdown(
            f'<div class="metric-container"><div class="metric-value" style="font-size:clamp(0.8rem, 2vw, 1rem);">{hw_info.platform}</div><div class="metric-label">Platform</div></div>',
            unsafe_allow_html=True)

    # Hardware optimizations
    with st.expander("Hardware Optimizations", expanded=False):
        optimizations = hardware_detector.get_recommended_params()

        col1, col2, col3 = st.columns(3)

        key_params_1 = {
            'chunk_size': 'Chunk Size',
            'embedding_batch_size': 'Batch Size',
            'max_workers': 'Workers'
        }

        key_params_2 = {
            'max_context_length': 'Context Length',
            'similarity_top_k': 'Top-K',
            'final_top_k': 'Final Top-K'
        }

        key_params_3 = {
            'temperature': 'Temperature',
            'safety_boost_factor': 'Safety Boost',
            'min_safety_score': 'Min Safety'
        }

        with col1:
            st.markdown('<div class="small-text">', unsafe_allow_html=True)
            st.markdown("**Core Parameters:**")
            for param, label in key_params_1.items():
                if param in optimizations:
                    st.text(f"{label}: {optimizations[param]}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="small-text">', unsafe_allow_html=True)
            st.markdown("**Retrieval Parameters:**")
            for param, label in key_params_2.items():
                if param in optimizations:
                    st.text(f"{label}: {optimizations[param]}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="small-text">', unsafe_allow_html=True)
            st.markdown("**Advanced Parameters:**")
            for param, label in key_params_3.items():
                if param in optimizations:
                    st.text(f"{label}: {optimizations[param]}")
            st.markdown('</div>', unsafe_allow_html=True)


def display_professional_system_status():
    """Professional system status display"""
    if st.session_state.system_initialized and st.session_state.rag_system and st.session_state.index_built:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            num_chunks = len(st.session_state.rag_system.vector_store.chunks)
            st.markdown(
                f'<div class="metric-container"><div class="metric-value">{num_chunks}</div><div class="metric-label">Document Chunks</div></div>',
                unsafe_allow_html=True)

        with col2:
            cache_stats = st.session_state.rag_system.get_cache_stats()
            total_queries = cache_stats.get('total_queries', 0)
            st.markdown(
                f'<div class="metric-container"><div class="metric-value">{total_queries}</div><div class="metric-label">Total Queries</div></div>',
                unsafe_allow_html=True)

        with col3:
            hw_tier = st.session_state.hardware_detector.hardware_info.performance_tier
            st.markdown(
                f'<div class="metric-container"><div class="metric-value tier-{hw_tier}">{hw_tier}</div><div class="metric-label">Hardware Tier</div></div>',
                unsafe_allow_html=True)

        with col4:
            cache_hit_rate = cache_stats.get('hit_rate_percent', 0)
            st.markdown(
                f'<div class="metric-container"><div class="metric-value">{cache_hit_rate:.1f}%</div><div class="metric-label">Cache Hit Rate</div></div>',
                unsafe_allow_html=True)

        with col5:
            config = st.session_state.config
            features_enabled = sum([
                config.enable_safety_detection,
                config.enable_safety_boosting,
                config.enable_evaluation
            ])
            st.markdown(
                f'<div class="metric-container"><div class="metric-value">{features_enabled}/3</div><div class="metric-label">Features Active</div></div>',
                unsafe_allow_html=True)

        # Safety analysis
        safety_stats = st.session_state.rag_system.get_safety_statistics()
        if 'safety_distribution' in safety_stats:
            st.markdown("#### Safety Analysis")
            dist = safety_stats['safety_distribution']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f'<div class="metric-container"><div class="metric-value safety-HIGH">{dist["HIGH"]}</div><div class="metric-label">HIGH Safety</div></div>',
                    unsafe_allow_html=True)
            with col2:
                st.markdown(
                    f'<div class="metric-container"><div class="metric-value safety-MEDIUM">{dist["MEDIUM"]}</div><div class="metric-label">MEDIUM Safety</div></div>',
                    unsafe_allow_html=True)
            with col3:
                st.markdown(
                    f'<div class="metric-container"><div class="metric-value safety-LOW">{dist["LOW"]}</div><div class="metric-label">LOW Safety</div></div>',
                    unsafe_allow_html=True)
            with col4:
                total_chunks = sum(dist.values())
                safety_ratio = (dist["HIGH"] + dist["MEDIUM"]) / total_chunks if total_chunks > 0 else 0
                st.markdown(
                    f'<div class="metric-container"><div class="metric-value">{safety_ratio:.2%}</div><div class="metric-label">Safety Critical</div></div>',
                    unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'hardware_detector' not in st.session_state:
        st.session_state.hardware_detector = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'index_built' not in st.session_state:
        st.session_state.index_built = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'torch_warning_shown' not in st.session_state:
        st.session_state.torch_warning_shown = False
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'safety_stats' not in st.session_state:
        st.session_state.safety_stats = None


def check_pytorch_compatibility():
    """Check for PyTorch-Streamlit compatibility issues"""
    if st.session_state.torch_warning_shown:
        return

    try:
        import torch
        _ = torch.__version__
        st.session_state.torch_warning_shown = True
    except Exception as e:
        if not st.session_state.torch_warning_shown:
            st.sidebar.warning("PyTorch compatibility issue detected. System should still work normally.")
            st.session_state.torch_warning_shown = True


def initialize_rag_system():
    """Initialize the RAG system with progress tracking"""
    if st.session_state.system_initialized:
        return True

    with st.spinner("Detecting hardware capabilities..."):
        hardware_detector = HardwareDetector()
        st.session_state.hardware_detector = hardware_detector

    # Display hardware info
    display_professional_hardware_info(hardware_detector)

    with st.spinner("Loading configuration..."):
        config_file = "rag_config.json"
        if Path(config_file).exists():
            config = RAGConfig.load(config_file)
            st.success(f"Loaded configuration from {config_file}")
        else:
            config = RAGConfig()
            st.info("Created default configuration")

        # Apply hardware optimizations
        config.ensure_new_parameters()
        config.apply_hardware_optimizations(hardware_detector)
        config.save(config_file)
        st.session_state.config = config

        # Show enhanced features status
        features_status = []
        if config.enable_safety_detection:
            features_status.append("Safety Detection")
        if config.enable_safety_boosting:
            features_status.append("Safety Boosting")
        if config.enable_file_context:
            features_status.append("File-Aware Chunking")
        if config.enable_evaluation:
            features_status.append("Quality Evaluation")

        st.info(f"Enhanced Features: {', '.join(features_status)}")
        st.info(f"Chunking: Recursive ({config.chunk_size} tokens) with {config.chunk_overlap} overlap")

    with st.spinner("Initializing RAG system..."):
        rag_system = HardwareAdaptiveRAGSystem(config, hardware_detector)
        st.session_state.rag_system = rag_system

    st.session_state.system_initialized = True
    st.success("Enhanced RAG System initialized successfully!")
    return True


def build_index():
    """Build or load the document index"""
    rag_system = st.session_state.rag_system

    if not rag_system:
        st.error("RAG system not initialized")
        return False

    # Check for documents
    docs_path = Path(rag_system.config.docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(docs_path.glob("*.txt"))
    md_files = list(docs_path.glob("*.md"))
    total_files = len(txt_files) + len(md_files)

    st.info(f"Document scan: Found {total_files} files ({len(txt_files)} .txt, {len(md_files)} .md)")

    # Show files found
    if total_files > 0:
        with st.expander("Files in documents/ folder"):
            for file_path in txt_files + md_files:
                try:
                    size = file_path.stat().st_size
                    st.text(f"   • {file_path.name}: {size:,} bytes")
                except:
                    st.text(f"   • {file_path.name}: (unknown size)")

    if total_files == 0:
        st.warning(f"No documents found in {docs_path.absolute()}")
        st.info("Please add .txt or .md files to the documents directory")
        return False

    # Check if index exists
    data_path = Path(rag_system.config.data_dir)
    index_exists = (data_path / "embeddings.npy").exists() and (data_path / "chunks.json").exists()

    if index_exists:
        st.info("Existing index found - loading...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Loading existing index...")
            progress_bar.progress(50)

            success = rag_system.build_index(force_rebuild=False)

            if success:
                progress_bar.progress(100)
                status_text.text("Existing index loaded!")
                st.session_state.index_built = True

                num_chunks = len(rag_system.vector_store.chunks)
                st.success(f"Loaded existing index with {num_chunks} document chunks")

                safety_stats = rag_system.get_safety_statistics()
                if 'safety_distribution' in safety_stats:
                    dist = safety_stats['safety_distribution']
                    st.info(f"Safety Distribution: HIGH={dist['HIGH']}, MEDIUM={dist['MEDIUM']}, LOW={dist['LOW']}")

                st.info("To process new documents, use 'Rebuild Index'")
                return True
            else:
                status_text.text("Failed to load existing index")
                st.error("Loading failed. Try rebuilding the index.")
                return False

        except Exception as e:
            st.error(f"Error loading index: {e}")
            return False
    else:
        st.info("No existing index - building new index...")
        return build_index_forced()


def build_index_forced():
    """Build index with forced rebuild"""
    rag_system = st.session_state.rag_system

    if not rag_system:
        st.error("RAG system not initialized")
        return False

    docs_path = Path(rag_system.config.docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(docs_path.glob("*.txt"))
    md_files = list(docs_path.glob("*.md"))
    total_files = len(txt_files) + len(md_files)

    if total_files == 0:
        st.warning(f"No documents found in {docs_path.absolute()}")
        st.info("Please add .txt or .md files to the documents directory")
        return False

    st.info(f"Found {total_files} document files ({len(txt_files)} .txt, {len(md_files)} .md)")

    with st.expander("Files Found"):
        for file_path in txt_files + md_files:
            try:
                size = file_path.stat().st_size
                st.text(f"• {file_path.name}: {size:,} bytes")
            except:
                st.text(f"• {file_path.name}: (size unknown)")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Forcing rebuild - clearing existing index...")
        progress_bar.progress(10)

        status_text.text("Processing documents with recursive chunking...")
        progress_bar.progress(30)

        status_text.text("Analyzing safety-critical content...")
        progress_bar.progress(50)

        status_text.text("Generating embeddings...")
        progress_bar.progress(70)

        status_text.text("Building vector index...")
        progress_bar.progress(90)

        success = rag_system.build_index(force_rebuild=True)

        if success:
            progress_bar.progress(100)
            status_text.text("Index rebuilt successfully!")
            st.session_state.index_built = True

            num_chunks = len(rag_system.vector_store.chunks)
            st.success(f"New index created with {num_chunks} document chunks")

            safety_stats = rag_system.get_safety_statistics()
            if 'safety_distribution' in safety_stats:
                dist = safety_stats['safety_distribution']
                st.info(f"Safety Analysis: HIGH={dist['HIGH']}, MEDIUM={dist['MEDIUM']}, LOW={dist['LOW']}")

            if num_chunks > 0:
                sources = {}
                safety_by_source = {}

                for chunk in rag_system.vector_store.chunks:
                    source = chunk.source_file
                    sources[source] = sources.get(source, 0) + 1

                    if source not in safety_by_source:
                        safety_by_source[source] = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

                    criticality = getattr(chunk, 'safety_criticality', 'LOW')
                    safety_by_source[source][criticality] += 1

                st.write("**Chunks per document:**")
                for source, count in sources.items():
                    safety_dist = safety_by_source[source]
                    st.text(
                        f"   • {source}: {count} chunks (HIGH: {safety_dist['HIGH']}, MEDIUM: {safety_dist['MEDIUM']}, LOW: {safety_dist['LOW']})")

            return True
        else:
            status_text.text("Failed to rebuild index")
            st.error("Index rebuilding failed. Check the logs for details.")
            return False

    except Exception as e:
        st.error(f"Error rebuilding index: {e}")
        progress_bar.progress(0)
        status_text.text("Rebuild failed")
        return False


def display_query_interface():
    """Professional query interface"""
    rag_system = st.session_state.rag_system

    if not rag_system or not st.session_state.index_built:
        st.warning("Please initialize system and build index first")
        return

    st.markdown("## Ask Questions About Electrical Safety")

    # Status cards
    config = st.session_state.config
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        safety_status = "Enabled" if config.enable_safety_detection else "Disabled"
        safety_class = "status-success" if config.enable_safety_detection else "status-error"
        st.markdown(
            f'<div class="status-card">Safety Detection: <span class="{safety_class}">{safety_status}</span></div>',
            unsafe_allow_html=True)

    with status_col2:
        boost_status = "Enabled" if config.enable_safety_boosting else "Disabled"
        boost_class = "status-success" if config.enable_safety_boosting else "status-error"
        st.markdown(
            f'<div class="status-card">Safety Boosting: <span class="{boost_class}">{boost_status}</span></div>',
            unsafe_allow_html=True)

    with status_col3:
        eval_status = "Enabled" if config.enable_evaluation else "Disabled"
        eval_class = "status-success" if config.enable_evaluation else "status-error"
        st.markdown(f'<div class="status-card">Evaluation: <span class="{eval_class}">{eval_status}</span></div>',
                    unsafe_allow_html=True)

    with status_col4:
        cache_stats = rag_system.get_cache_stats()
        hit_rate = cache_stats.get('hit_rate_percent', 0)
        cache_class = "status-success" if hit_rate > 20 else "status-warning" if hit_rate > 5 else "status-error"
        st.markdown(
            f'<div class="status-card">Cache Hit Rate: <span class="{cache_class}">{hit_rate:.1f}%</span></div>',
            unsafe_allow_html=True)

    # Example queries
    with st.expander("Example Queries (Safety-Focused)", expanded=False):
        example_queries = [
    "What are the minimum approach distances for 138kV equipment?",
    "What PPE is required for HRC 2 electrical work?",
    "What is the acceptable contact resistance for circuit breaker testing?",
    "List the steps for lockout/tagout energy isolation",
    "What are the requirements for working alone in utility operations?",
    "What are the check-in times for working alone?"
    "Walk me through the confined space entry permit process",
    "What is the proper sequence for applying electrical grounds?",
    "What are the temperature thresholds for stopping work in prairie conditions?",
    "When is a Grid Control Center permit required?",
    "What are the fall protection requirements for transmission tower work?",
    "What are the pass/fail criteria for SF6 gas analysis in circuit breakers?",
    "What insulation resistance values indicate transformer failure?",
    "How do I calculate total fall clearance for fall protection systems?",
    "At what wind speed does work have to stop?",
    "My circuit breaker failed the timing testing, what should I check now?",
    "A transformer shows high power factor readings after a power factor test, what does this indicate?",
    "What gases in DGA analysis suggest internal arcing?",
    "What's the response protocol for arc flash incidents?",
    "How do I remove someone else's lockout device in an emergency?",
    "What are the rescue requirements for confined space work?",
    "Steps for emergency permit procedures during equipment failures?",
    "Compare the requirements for EAP vs EIP permits",
    "What's the difference between qualified and authorized employees?",
    "HRC 2 vs HRC 3 clothing - when do I use each?",
    "Grounding vs bonding - which procedure applies when?"
]

        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(query, key=f"example_query_{i}", use_container_width=True):
                    st.session_state.current_query = query

    # Query input
    st.markdown("### Enter Your Question")
    query = st.text_area(
        "Type your electrical safety question here:",
        value=st.session_state.get('current_query', ''),
        height=100,
        key="query_input",
        help="Ask about electrical safety procedures, PPE requirements, OSHA standards, arc flash protection, etc."
    )

    # Action buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    with btn_col1:
        if st.button("Search", type="primary", disabled=not query.strip(), key="search_btn",
                     use_container_width=True):
            st.session_state.pending_query = query.strip()
            st.session_state.trigger_search = True

    with btn_col2:
        if st.button("Clear History", key="clear_history_btn", use_container_width=True):
            st.session_state.query_history = []
            st.rerun()

    with btn_col3:
        if st.button("Quick Benchmark", key="benchmark_btn", use_container_width=True):
            run_enhanced_benchmark()

    with btn_col4:
        if st.button("Run Evaluation", key="evaluate_btn", use_container_width=True):
            run_system_evaluation()

    # Process query outside column structure
    if st.session_state.get('trigger_search', False) and st.session_state.get('pending_query'):
        st.session_state.trigger_search = False
        query_to_process = st.session_state.pending_query
        st.session_state.pending_query = None
        process_query_outside_columns(query_to_process)


def process_query_outside_columns(query: str):
    """Process query and display results at main page level"""
    if not query:
        return

    rag_system = st.session_state.rag_system
    start_time = time.time()

    st.markdown("---")

    with st.spinner("Processing your question..."):
        result = rag_system.query(query)

    query_time = time.time() - start_time

    # Add to history
    st.session_state.query_history.append({
        'query': query,
        'result': result,
        'timestamp': time.strftime('%H:%M:%S'),
        'processing_time': query_time
    })

    display_professional_query_result(result, query_time)


def display_professional_query_result(result: Dict[str, Any], query_time: float):
    """Professional query result display"""
    if 'error' in result:
        st.error(f"Error: {result['error']}")
        return

    # Performance metrics
    st.markdown("## Query Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    cache_status = "Cached" if result.get('cached', False) else "Fresh"
    tier = result.get('hardware_tier', 'UNKNOWN')
    sources_count = result.get('num_chunks_retrieved', 0)
    safety_boost = result.get('safety_boosted', False)

    with col1:
        st.metric("Response", cache_status)

    with col2:
        st.metric("Hardware", tier)

    with col3:
        st.metric("Time", f"{query_time:.2f}s")

    with col4:
        st.metric("Sources", sources_count)

    with col5:
        st.metric("Mode", "Boosted" if safety_boost else "Standard")

    st.divider()

    # Answer section
    st.markdown("### Answer")

    answer_text = result.get('answer', 'No answer provided')
    if len(answer_text) > 2000:
        answer_text = answer_text[:2000] + "..."

    st.info(answer_text)

    # Sources section
    if result.get('sources'):
        st.markdown("### Sources")

        for i, source in enumerate(result['sources'], 1):
            safety_info = ""
            if 'safety_criticality' in source:
                criticality = source['safety_criticality']
                safety_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(criticality, "⚪")
                safety_info = f" {safety_color} {criticality}"

            header_text = f"Source {i}: {source['source_file']} (Similarity: {source['similarity_score']:.3f}){safety_info}"

            with st.expander(header_text):
                # Get full content without truncation
                content_preview = source['content_preview']
                
                # Remove any existing truncation suffix
                if content_preview.endswith('...'):
                    content_preview = content_preview[:-3]
                
                # Display full content in scrollable text area
                st.text_area(
                    "Full Chunk Content:",
                    value=content_preview,
                    height=300,
                    disabled=True,
                    key=f"chunk_content_{i}"
                )

                if 'safety_criticality' in source:
                    st.write(
                        f"**Metadata:** Chunk ID: {source['chunk_id']} | Similarity: {source['similarity_score']:.3f} | Safety Level: {source['safety_criticality']} | Safety Score: {source.get('safety_score', 0):.2f}")

                    if source.get('safety_keywords'):
                        keywords = ', '.join(source['safety_keywords'][:8])
                        st.write(f"**Safety Keywords:** *{keywords}*")

    # Performance details
    with st.expander("Performance Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Query Performance**")
            st.write(f"Processing Time: {query_time:.3f}s")
            st.write(f"Chunks Retrieved: {result.get('num_chunks_retrieved', 0)}")
            st.write(f"Hardware Tier: {result.get('hardware_tier', 'UNKNOWN')}")
            st.write(f"Prompt Length: {result.get('prompt_length', 0)} chars")

        with col2:
            st.write("**Enhanced Features**")
            st.write(f"Cached: {'Yes' if result.get('cached', False) else 'No'}")
            st.write(f"Safety Boosted: {'Yes' if result.get('safety_boosted', False) else 'No'}")
            st.write("Quality: Enhanced")

        if result.get('sources'):
            st.write("**Safety Analysis**")
            safety_dist = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for source in result['sources']:
                criticality = source.get('safety_criticality', 'LOW')
                safety_dist[criticality] += 1

            total = sum(safety_dist.values())
            critical_pct = (safety_dist['HIGH'] + safety_dist['MEDIUM']) / total * 100 if total > 0 else 0

            st.write(f"HIGH: {safety_dist['HIGH']} | MEDIUM: {safety_dist['MEDIUM']} | LOW: {safety_dist['LOW']}")
            st.write(f"Critical Content: {critical_pct:.1f}%")

def run_enhanced_benchmark():
    """Run performance benchmark"""
    rag_system = st.session_state.rag_system

    test_queries = [
        "What is electrical safety?",
        "How do lockout tagout procedures work?",
        "What PPE is required for arc flash protection?",
        "Explain electrical grounding procedures",
        "What are switching operation safety requirements?",
        "How do I calculate incident energy for arc flash?",
        "What voltage levels require qualified persons?",
        "What are the OSHA electrical safety standards?"
    ]

    st.subheader("Performance Benchmark")

    results = []
    progress_bar = st.progress(0)

    for i, test_query in enumerate(test_queries):
        with st.spinner(f"Testing query {i + 1}/{len(test_queries)}..."):
            start_time = time.time()
            result = rag_system.query(test_query)
            query_time = time.time() - start_time

            # Calculate safety metrics
            safety_high = safety_medium = safety_low = 0

            if result.get('sources'):
                for source in result['sources']:
                    criticality = source.get('safety_criticality', 'LOW')
                    if criticality == 'HIGH':
                        safety_high += 1
                    elif criticality == 'MEDIUM':
                        safety_medium += 1
                    else:
                        safety_low += 1

            results.append({
                'Query': test_query[:40] + "..." if len(test_query) > 40 else test_query,
                'Time (s)': round(query_time, 3),
                'Cached': result.get('cached', False),
                'Sources': result.get('num_chunks_retrieved', 0),
                'Safety Boosted': result.get('safety_boosted', False),
                'High Safety': safety_high,
                'Med Safety': safety_medium,
                'Low Safety': safety_low,
                'Hardware Tier': result.get('hardware_tier', 'UNKNOWN')
            })

            progress_bar.progress((i + 1) / len(test_queries))

    # Display results
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # Summary statistics
    avg_time = df['Time (s)'].mean()
    cached_count = df['Cached'].sum()
    boosted_count = df['Safety Boosted'].sum()
    total_high_safety = df['High Safety'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Time", f"{avg_time:.3f}s")
    with col2:
        st.metric("Cache Hits", f"{cached_count}/{len(test_queries)}")
    with col3:
        st.metric("Safety Boosted", f"{boosted_count}/{len(test_queries)}")
    with col4:
        st.metric("High Safety Sources", total_high_safety)

    # Safety distribution chart
    safety_data = {
        'Safety Level': ['HIGH', 'MEDIUM', 'LOW'],
        'Count': [df['High Safety'].sum(), df['Med Safety'].sum(), df['Low Safety'].sum()]
    }

    fig = px.bar(safety_data, x='Safety Level', y='Count',
                 title="Safety Distribution in Benchmark Results",
                 color='Safety Level',
                 color_discrete_map={'HIGH': '#e53e3e', 'MEDIUM': '#ed8936', 'LOW': '#38a169'})
    st.plotly_chart(fig, use_container_width=True)


def run_system_evaluation():
    """Run comprehensive system evaluation"""
    rag_system = st.session_state.rag_system

    if not rag_system.evaluator:
        st.warning("Evaluation not enabled in configuration")
        return

    st.subheader("System Evaluation")

    with st.spinner("Running comprehensive evaluation..."):
        eval_results = rag_system.evaluate_system()

    if 'evaluation_disabled' in eval_results:
        st.error("Evaluation is disabled")
        return

    st.session_state.evaluation_results = eval_results

    # Display evaluation results
    st.markdown("### Retrieval Quality Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        relevance_at_k = eval_results.get('relevance_at_k_mean', 0)
        relevance_std = eval_results.get('relevance_at_k_std', 0)
        st.metric(
            f"Relevance@{st.session_state.config.similarity_top_k}",
            f"{relevance_at_k:.3f} ± {relevance_std:.3f}",
            help="% of retrieved chunks that contain expected keywords"
        )

    with col2:
        coverage_at_k = eval_results.get('coverage_at_k_mean', 0)
        coverage_std = eval_results.get('coverage_at_k_std', 0)
        st.metric(
            f"Coverage@{st.session_state.config.similarity_top_k}",
            f"{coverage_at_k:.3f} ± {coverage_std:.3f}",
            help="% of queries that found at least 1 relevant chunk"
        )

    with col3:
        keyword_coverage = eval_results.get('keyword_coverage_mean', 0)
        keyword_std = eval_results.get('keyword_coverage_std', 0)
        st.metric(
            "Keyword Coverage",
            f"{keyword_coverage:.3f} ± {keyword_std:.3f}",
            help="% of expected keywords found in top-K results"
        )

    with col4:
        if st.session_state.config.enable_safety_boosting:
            safety_prioritization = eval_results.get('safety_prioritization_mean', 0)
            safety_std = eval_results.get('safety_prioritization_std', 0)
            st.metric(
                "Safety Prioritization",
                f"{safety_prioritization:.3f} ± {safety_std:.3f}",
                help="How well safety-critical content is ranked higher"
            )
        else:
            st.metric("Safety Prioritization", "Disabled")

    # Performance metrics
    st.markdown("### Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        response_time = eval_results.get('response_times_mean', 0)
        response_std = eval_results.get('response_times_std', 0)
        st.metric(
            "Avg Response Time",
            f"{response_time:.3f}s ± {response_std:.3f}s"
        )

    with col2:
        chunks_retrieved = eval_results.get('chunks_retrieved_mean', 0)
        st.metric("Avg Chunks Retrieved", f"{chunks_retrieved:.1f}")

    with col3:
        queries_evaluated = eval_results.get('queries_evaluated', 0)
        st.metric("Queries Evaluated", queries_evaluated)

    with col4:
        total_time = eval_results.get('total_evaluation_time', 0)
        st.metric("Total Eval Time", f"{total_time:.2f}s")

    # Quality checks
    if 'quality_checks' in eval_results:
        st.markdown("### Quality Checks")
        checks = eval_results['quality_checks']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            passed = checks.get('good_relevance', False)
            st.markdown(f"**Good Relevance:** {'✅ PASS' if passed else '❌ FAIL'}")
        with col2:
            passed = checks.get('good_coverage', False)  
            st.markdown(f"**Good Coverage:** {'✅ PASS' if passed else '❌ FAIL'}")
        with col3:
            passed = checks.get('good_keywords', False)
            st.markdown(f"**Good Keywords:** {'✅ PASS' if passed else '❌ FAIL'}")
        with col4:
            passed = checks.get('fast_response', False)
            st.markdown(f"**Fast Response:** {'✅ PASS' if passed else '❌ FAIL'}")

        # Overall status  
        passed_checks = sum(1 for passed in checks.values() if passed)
        total_checks = len(checks)
        
        if passed_checks == total_checks:
            st.success(f"All quality checks passed! ({passed_checks}/{total_checks})")
        else:
            st.warning(f"Quality checks: {passed_checks}/{total_checks} passed")

    # Performance visualization
    st.markdown("### Performance Visualization")

    config = st.session_state.config
    metrics_data = {
        'Metric': [
            f'Relevance@{config.similarity_top_k}',
            f'Coverage@{config.similarity_top_k}',
            'Keyword Coverage',
            'Speed Score'
        ],
        'Score': [
            eval_results.get('relevance_at_k_mean', 0),
            eval_results.get('coverage_at_k_mean', 0),
            eval_results.get('keyword_coverage_mean', 0),
            max(0, 1 - (eval_results.get('response_times_mean', 0) / config.max_response_time))
        ],
        'Target': [
            config.min_relevance_at_k,
            config.min_coverage_at_k,
            config.min_keyword_coverage,
            0.8
        ]
    }

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=metrics_data['Score'],
        theta=metrics_data['Metric'],
        fill='toself',
        name='Actual Performance',
        line_color='blue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=metrics_data['Target'],
        theta=metrics_data['Metric'],
        fill='toself',
        name='Target Performance',
        line_color='red',
        opacity=0.6
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="System Performance vs Targets"
    )

    st.plotly_chart(fig, use_container_width=True)


def display_professional_system_statistics():
    """Display system statistics with professional styling"""
    rag_system = st.session_state.rag_system

    if not rag_system:
        st.warning("System not initialized")
        return

    st.subheader("System Statistics")

    # Cache statistics
    cache_stats = rag_system.get_cache_stats()

    if 'caching_disabled' not in cache_stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", cache_stats.get('total_queries', 0))
        with col2:
            st.metric("Cache Hits", cache_stats.get('hits', 0))
        with col3:
            hit_rate = cache_stats.get('hit_rate_percent', 0)
            st.metric("Hit Rate", f"{hit_rate:.1f}%")
        with col4:
            st.metric("Cached Queries", cache_stats.get('cached_queries', 0))

    # Safety statistics
    safety_stats = rag_system.get_safety_statistics()
    if 'index_not_loaded' not in safety_stats:
        st.subheader("Safety Analysis Statistics")

        col1, col2, col3, col4 = st.columns(4)

        dist = safety_stats['safety_distribution']
        with col1:
            st.metric("HIGH Safety Chunks", dist['HIGH'])
        with col2:
            st.metric("MEDIUM Safety Chunks", dist['MEDIUM'])
        with col3:
            st.metric("LOW Safety Chunks", dist['LOW'])
        with col4:
            avg_safety = safety_stats['average_safety_score']
            st.metric("Avg Safety Score", f"{avg_safety:.2f}")

        # Safety distribution chart
        safety_chart_data = {
            'Safety Level': ['HIGH', 'MEDIUM', 'LOW'],
            'Count': [dist['HIGH'], dist['MEDIUM'], dist['LOW']]
        }

        fig = px.pie(safety_chart_data, values='Count', names='Safety Level',
                     title="Safety Criticality Distribution",
                     color_discrete_map={'HIGH': '#e53e3e', 'MEDIUM': '#ed8936', 'LOW': '#38a169'})
        st.plotly_chart(fig, use_container_width=True)

        # Top safety keywords
        if safety_stats.get('top_safety_keywords'):
            st.subheader("Top Safety Keywords")
            keywords_df = pd.DataFrame(
                safety_stats['top_safety_keywords'][:10],
                columns=['Keyword', 'Frequency']
            )

            fig = px.bar(keywords_df, x='Frequency', y='Keyword', orientation='h',
                         title="Most Frequent Safety Keywords")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # Vector store information
    if rag_system.vector_store.is_loaded:
        st.subheader("Document Index")

        num_chunks = len(rag_system.vector_store.chunks)
        if num_chunks > 0:
            # Document source analysis
            sources = {}
            safety_by_source = {}

            for chunk in rag_system.vector_store.chunks:
                source = chunk.source_file
                sources[source] = sources.get(source, 0) + 1

                if source not in safety_by_source:
                    safety_by_source[source] = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}

                criticality = getattr(chunk, 'safety_criticality', 'LOW')
                safety_by_source[source][criticality] += 1

            # Source analysis table
            source_data = []
            for source, count in sources.items():
                safety_dist = safety_by_source[source]
                source_data.append({
                    'Document': source,
                    'Total Chunks': count,
                    'HIGH Safety': safety_dist['HIGH'],
                    'MEDIUM Safety': safety_dist['MEDIUM'],
                    'LOW Safety': safety_dist['LOW'],
                    'Safety Ratio': (safety_dist['HIGH'] + safety_dist['MEDIUM']) / count
                })

            source_df = pd.DataFrame(source_data)
            st.dataframe(source_df, use_container_width=True)

            # Stacked bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='HIGH Safety',
                x=source_df['Document'],
                y=source_df['HIGH Safety'],
                marker_color='#e53e3e'
            ))

            fig.add_trace(go.Bar(
                name='MEDIUM Safety',
                x=source_df['Document'],
                y=source_df['MEDIUM Safety'],
                marker_color='#ed8936'
            ))

            fig.add_trace(go.Bar(
                name='LOW Safety',
                x=source_df['Document'],
                y=source_df['LOW Safety'],
                marker_color='#38a169'
            ))

            fig.update_layout(
                barmode='stack',
                title='Document Chunks by Safety Level',
                xaxis_title='Document',
                yaxis_title='Number of Chunks'
            )

            st.plotly_chart(fig, use_container_width=True)


def display_professional_query_history():
    """Display query history with professional styling"""
    if not st.session_state.query_history:
        st.info("No queries yet. Ask a question to see history here.")
        return

    st.subheader("Query History")

    # Summary statistics
    if len(st.session_state.query_history) > 0:
        total_queries = len(st.session_state.query_history)
        cached_queries = sum(1 for entry in st.session_state.query_history
                             if entry['result'].get('cached', False))
        boosted_queries = sum(1 for entry in st.session_state.query_history
                              if entry['result'].get('safety_boosted', False))
        avg_time = sum(entry['processing_time'] for entry in st.session_state.query_history) / total_queries

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Cached", f"{cached_queries}/{total_queries}")
        with col3:
            st.metric("Safety Boosted", f"{boosted_queries}/{total_queries}")
        with col4:
            st.metric("Avg Time", f"{avg_time:.2f}s")

    # Display history items
    for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):
        result = entry['result']

        # Professional header
        cache_status = "Cached" if result.get('cached', False) else "Fresh"
        safety_status = "Boosted" if result.get('safety_boosted', False) else "Standard"
        header = f"[{entry['timestamp']}] {entry['query'][:50]}..."

        with st.expander(header):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown('<div class="small-text">', unsafe_allow_html=True)
                st.markdown("**Answer:**")
                answer_text = result.get('answer', 'No answer')
                if len(answer_text) > 300:
                    answer_text = answer_text[:300] + "..."

                # Format answer with theme-compatible styling
                formatted_answer = answer_text.replace('\n', '<br/>')
                st.markdown(
                    f'<div class="small-text" style="padding: 0.5rem; background-color: var(--bg-secondary); border-radius: 4px; margin: 0.25rem 0; color: var(--text-secondary);">{formatted_answer}</div>',
                    unsafe_allow_html=True)

                if result.get('sources'):
                    st.markdown("**Sources:**")
                    for j, source in enumerate(result['sources'][:3], 1):
                        safety_info = ""
                        if 'safety_criticality' in source:
                            criticality = source['safety_criticality']
                            safety_info = f" ({criticality})"

                        source_text = f"{j}. {source['source_file']} (Score: {source['similarity_score']:.3f}){safety_info}"
                        st.text(source_text)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="tiny-text">', unsafe_allow_html=True)
                st.markdown("**Performance:**")
                st.text(f"Time: {entry['processing_time']:.3f}s")
                st.text(f"Response: {cache_status}")
                st.text(f"Mode: {safety_status}")
                st.text(f"Sources: {result.get('num_chunks_retrieved', 0)}")
                st.text(f"Tier: {result.get('hardware_tier', 'UNKNOWN')}")

                # Safety distribution
                if result.get('sources'):
                    safety_dist = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
                    for source in result['sources']:
                        criticality = source.get('safety_criticality', 'LOW')
                        safety_dist[criticality] += 1

                    st.text(f"Safety Distribution:")
                    st.text(f"  HIGH: {safety_dist['HIGH']}")
                    st.text(f"  MED: {safety_dist['MEDIUM']}")
                    st.text(f"  LOW: {safety_dist['LOW']}")
                st.markdown('</div>', unsafe_allow_html=True)


def display_professional_configuration():
    """Display system configuration with professional styling"""
    config = st.session_state.config

    if not config:
        st.warning("Configuration not loaded")
        return

    st.subheader("System Configuration")

    # Group settings in expandable sections
    with st.expander("Model & Basic Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Embedding Model: {config.embedding_model}")
            st.text(f"LLM Model: {config.ollama_model}")
            st.text(f"Hardware Optimized: {config.hardware_optimized}")
            st.text(f"Hardware Tier: {config.hardware_tier}")
        with col2:
            st.text(f"Parallel Init: {config.parallel_init}")
            st.text(f"Warm Models: {config.warm_models_on_startup}")
            st.text(f"Cache Enabled: {config.enable_cache}")
            st.text(f"Data Loading Optimized: {config.optimize_data_loading}")

    with st.expander("Chunking Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Primary Chunk Size: {config.chunk_size} tokens")
            st.text(f"Alternative Chunk Size: {config.chunk_size_alt} tokens")
            st.text(f"Primary Overlap: {config.chunk_overlap} tokens")
            st.text(f"Alternative Overlap: {config.chunk_overlap_alt} tokens")
        with col2:
            st.text(f"Min Chunk Size: {config.min_chunk_size} tokens")
            st.text(f"Embedding Batch Size: {config.embedding_batch_size}")
            st.text(f"Max Workers: {config.max_workers}")
            st.text(f"Use Model Cache: {config.use_model_cache}")

    with st.expander("Safety Detection Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Safety Detection: {config.enable_safety_detection}")
            st.text(f"Safety Boosting: {config.enable_safety_boosting}")
            st.text(f"Safety Boost Factor: {config.safety_boost_factor}")
        with col2:
            st.text(f"Min Safety Score: {config.min_safety_score}")

    with st.expander("Retrieval Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Similarity Top-K: {config.similarity_top_k}")
            st.text(f"Final Top-K: {config.final_top_k}")
            st.text(f"Similarity Threshold: {config.similarity_threshold}")
        with col2:
            st.text(f"Max Context Length: {config.max_context_length}")
            st.text(f"Temperature: {config.temperature}")

    with st.expander("Evaluation Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Evaluation Enabled: {config.enable_evaluation}")
            st.text(f"Test Size: {config.eval_test_size}")
        with col2:
            st.text(f"Min Relevance@K: {config.min_relevance_at_k}")
            st.text(f"Min Coverage@K: {config.min_coverage_at_k}")

        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Min Keyword Coverage: {config.min_keyword_coverage}")
        with col2:
            st.text(f"Max Response Time: {config.max_response_time}s")

    if st.button("Reload Configuration", key="reload_config_btn"):
        try:
            new_config = RAGConfig.load("rag_config.json")
            new_config.apply_hardware_optimizations(st.session_state.hardware_detector)
            st.session_state.config = new_config
            st.success("Configuration reloaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to reload configuration: {e}")


def main():
    """Main Streamlit application"""
    init_session_state()
    check_pytorch_compatibility()

    # Professional header
    st.markdown("# Electrical Utility RAG System")
    st.markdown(
        '<div class="tiny-text"><em>Hardware-Adaptive Knowledge Assistant with Safety Detection & Quality Evaluation</em></div>',
        unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.markdown("### Navigation")

    # System status
    if st.session_state.system_initialized:
        st.sidebar.success("System Ready")
        if st.session_state.config:
            hw_tier = st.session_state.hardware_detector.hardware_info.performance_tier
            tier_class = f"tier-{hw_tier}"
            st.sidebar.markdown(f'<div class="tiny-text">Hardware: <span class="{tier_class}">{hw_tier}</span></div>',
                                unsafe_allow_html=True)

            safety_status = "Enabled" if st.session_state.config.enable_safety_detection else "Disabled"
            eval_status = "Enabled" if st.session_state.config.enable_evaluation else "Disabled"
            st.sidebar.markdown(f'<div class="tiny-text">Safety: {safety_status} | Eval: {eval_status}</div>',
                                unsafe_allow_html=True)
    else:
        st.sidebar.warning("System Not Initialized")

    if st.session_state.index_built and st.session_state.rag_system:
        st.sidebar.success("Index Ready")
        num_chunks = len(st.session_state.rag_system.vector_store.chunks)

        safety_stats = st.session_state.rag_system.get_safety_statistics()
        if 'safety_distribution' in safety_stats:
            dist = safety_stats['safety_distribution']
            st.sidebar.markdown(
                f'<div class="tiny-text">Chunks: {num_chunks}<br/>HIGH: {dist["HIGH"]}, MED: {dist["MEDIUM"]}, LOW: {dist["LOW"]}</div>',
                unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f'<div class="tiny-text">Chunks: {num_chunks}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.warning("Index Not Built")

    # Navigation
    tab = st.sidebar.radio(
        "Select Page:",
        ["Home", "Query", "Statistics", "History", "Evaluation", "Configuration"],
        key="nav_tab"
    )

    # Control buttons
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Controls")

    if not st.session_state.system_initialized:
        if st.sidebar.button("Initialize System", type="primary"):
            initialize_rag_system()
            st.rerun()

    if st.session_state.system_initialized:
        if not st.session_state.index_built:
            if st.sidebar.button("Build Index", type="primary"):
                if build_index():
                    st.rerun()
        else:
            if st.sidebar.button("Rebuild Index"):
                st.session_state.index_built = False
                if build_index_forced():
                    st.rerun()

        if st.sidebar.button("Clear Cache"):
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_cache()
                st.sidebar.success("Cache cleared!")

    # Main content
    if tab == "Home":
        if not st.session_state.system_initialized:
            st.info("Welcome! Please initialize the system to get started.")

            st.markdown("#### Enhanced Features:")
            st.markdown("""
            <div class="small-text">
            • <strong>Recursive Chunking:</strong> Smart text splitting preserving context<br/>
            • <strong>Safety Detection:</strong> Automatic identification of safety-critical content<br/>
            • <strong>Safety Boosting:</strong> Prioritizes safety-relevant information<br/>
            • <strong>Quality Evaluation:</strong> Automated assessment with transparent metrics<br/>
            • <strong>Hardware Adaptive:</strong> Optimizes performance for your system
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("System ready! Query electrical safety documents with enhanced features.")
            display_professional_system_status()

            if st.session_state.hardware_detector:
                display_professional_hardware_info(st.session_state.hardware_detector)

    elif tab == "Query":
        display_query_interface()

    elif tab == "Statistics":
        display_professional_system_statistics()

    elif tab == "History":
        display_professional_query_history()

    elif tab == "Evaluation":
        if st.session_state.rag_system and st.session_state.index_built:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Run Full Evaluation", type="primary"):
                    run_system_evaluation()
            with col2:
                if st.button("Quick Benchmark"):
                    run_enhanced_benchmark()
        else:
            st.warning("Please initialize system and build index first")

    elif tab == "Configuration":
        display_professional_configuration()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="tiny-text"><em>RAG System v2.0<br/>Safety-Aware • Hardware-Adaptive</em></div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()