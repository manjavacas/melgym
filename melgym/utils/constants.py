import os

CLEAN_COMMAND = 'clean'
CLEAN_ALL_COMMAND = 'cleanall'

ROOT_DIR = os.getcwd()

DATA_DIR = os.path.join(ROOT_DIR, 'melgym', 'data')
EXEC_DIR = os.path.join(ROOT_DIR, 'melgym', 'exec')

MELGEN_PATH = os.path.join(EXEC_DIR, 'melgen-fusion-186_bdba')
MELCOR_PATH = os.path.join(EXEC_DIR, 'melcor-fusion-186_bdba')