# -- Project information -----------------------------------------------------

import sys
import os
import melgym

project = 'MELGYM'
copyright = '2025, Antonio Manjavacas'
author = 'Antonio Manjavacas'
release = '2.0.0'

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
html_logo = '_static/images/logo-2.png'

html_static_path = [
    '_static'
]

html_css_files = [
    'css/custom.css',
]

