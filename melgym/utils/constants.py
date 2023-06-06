
import os

################################## PATHS ##################################

ROOT_DIR = os.getcwd()

DATA_DIR = os.path.join(ROOT_DIR, 'melgym', 'data')
EXEC_DIR = os.path.join(ROOT_DIR, 'melgym', 'exec')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'melgym', 'out')

INPUT_PATH = os.path.join(DATA_DIR, 'hvac.inp')

MELIN_PATH = os.path.join(OUTPUT_DIR, 'MELIN')

MELGEN_PATH = os.path.join(EXEC_DIR, 'MELGEN')
MELCOR_PATH = os.path.join(EXEC_DIR, 'MELCOR')

############################### HVAC SPECS ###############################

MAX_DUCT_VELOCITY = 10.

N_HVAC_BRANCHES = 1
N_HVAC_SERVED_ROOMS = 5