"""Build the reference from the same configurations used by simulations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))
project = "TipBox"
author = "Juan Nathaniel"
copyright = "2026, Juan Nathaniel"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "model_parameters",
]
html_theme = "furo"
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/repository.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}
html_title = "TipBox"
html_logo = "tipbox-icon-300ppi.png"
html_baseurl = "https://juannat7.github.io/tipbox/"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#216a76",
        "color-brand-content": "#216a76",
    },
    "dark_css_variables": {
        "color-brand-primary": "#75c3ce",
        "color-brand-content": "#75c3ce",
    },
}
html_static_path = ["_static"]
html_css_files = ["tipbox.css"]
html_extra_path = ["llms.txt"]
exclude_patterns = ["_build"]
autodoc_typehints = "description"
