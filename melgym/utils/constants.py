"""
Constants used by MELGYM.

- `BASE_DIR`: The base directory of the project.
- `OUTPUT_DIR`: The directory where output files are stored.
- `EXEC_DIR`: The directory where executable files are stored.
- `MELGEN_PATH`: The path to the MELGEN executable.
- `MELCOR_PATH`: The path to the MELCOR executable.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "out")

EXEC_DIR = os.path.join(BASE_DIR, "exec")
MELGEN_PATH = os.path.join(EXEC_DIR, "MELGEN")
MELCOR_PATH = os.path.join(EXEC_DIR, "MELCOR")

if not os.path.isfile(MELGEN_PATH):
    raise FileNotFoundError(f"MELGEN executable not found at {MELGEN_PATH}")

if not os.path.isfile(MELCOR_PATH):
    raise FileNotFoundError(f"MELCOR executable not found at {MELCOR_PATH}")
