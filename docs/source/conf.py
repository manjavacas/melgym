# -- Project information -----------------------------------------------------

import sys
import os

project = 'MELGYM'
copyright = '2024, Antonio Manjavacas'
author = 'Antonio Manjavacas'
release = '0.1'

# -- General configuration ---------------------------------------------------

sys.path.insert(0, os.path.abspath('../../'))

extensions = ['sphinx_rtd_theme',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
