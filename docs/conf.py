"""Build the reference from the same configurations used by simulations."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))
project = "TipBox"
author = "Juan Nathaniel"
copyright = "2026, Juan Nathaniel"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "model_parameters",
]
html_theme = "alabaster"
html_title = "TipBox · Tipping-point benchmarks"
html_theme_options = {
    "description": "Box models and classical early warning signals",
    "fixed_sidebar": True,
    "page_width": "1120px",
    "sidebar_width": "240px",
}
html_static_path = ["_static"]
html_css_files = ["tipbox.css"]
exclude_patterns = ["_build"]
autodoc_typehints = "description"
