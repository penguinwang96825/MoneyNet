import os
import streamlit as st
from pathlib import Path


def hide_header_and_footer():
    hide_streamlit_style = """
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def read_markdown_file(markdown_file):
    """
    Examples
    --------
    >>> markdown = read_markdown_file("introduction.md")
    >>> st.markdown(markdown, unsafe_allow_html=True)
    """
    return Path(markdown_file).read_text(encoding='utf-8')


def read_css_file(css_file):
    """
    Examples
    --------
    >>> css_file = read_markdown_file("main.css")
    >>> st.markdown(css_file, unsafe_allow_html=True)
    """
    return f"<style>{Path(css_file).read_text(encoding='utf-8')}</style>"