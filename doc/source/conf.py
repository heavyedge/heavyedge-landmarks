# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HeavyEdge-Landmarks"
copyright = "2025, Jisoo Song"
author = "Jisoo Song"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False

intersphinx_mapping = {
    "heavyedge": ("https://heavyedge.readthedocs.io/en/latest/", None),
}

numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

plot_include_source = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "HeavyEdge-Dataset",
    },
    "show_toc_level": 2,
}

plot_html_show_formats = False
plot_html_show_source_link = False
