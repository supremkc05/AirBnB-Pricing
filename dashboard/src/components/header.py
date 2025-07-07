"""
Header component for the Airbnb Price Analytics Dashboard.
"""

import streamlit as st
from ..config.settings import CSS_STYLES, PAGE_CONFIG


def render_header():
    """Render the main header with styling."""
    # Apply CSS styles
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    # Main header
    st.markdown(
        f'<h1 class="main-header">{PAGE_CONFIG["icon"]} Airbnb Price Analytics Dashboard</h1>', 
        unsafe_allow_html=True
    )


def render_page_header(page_title, icon="📊"):
    """Render a page-specific header."""
    st.markdown(
        f'<h1 class="main-header">{icon} {page_title}</h1>', 
        unsafe_allow_html=True
    )


def render_section_header(section_title):
    """Render a section header."""
    st.markdown(
        f'<h3 class="section-header">{section_title}</h3>', 
        unsafe_allow_html=True
    )


def render_footer():
    """Render the footer."""
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.9rem;'>"
        "🏠 Airbnb Price Analytics Dashboard | Built with Streamlit & Python"
        "</p>",
        unsafe_allow_html=True
    )
