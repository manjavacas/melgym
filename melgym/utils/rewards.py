import numpy as np

def get_sgs_reward(obs, air_percent):
    '''
    Checks how many CVs have an air concentration lower to a given value.
    '''
    return sum(x for x in obs if x < air_percent)


def get_sgs_reward_2(obs, air_percent):
    '''
    Check the global distance from CVs air concentration to a given value.
    '''
    return sum(air_percent - x for x in obs)

def get_sgs_reward_3(obs):
    '''
    Uses the negative observation value as reward value.
    '''
    return -sum(x for x in obs)

# TO-DO: recorrer presiones y ver cuantas tienen una variacion < K
# Otra opcion: sumar las variaciones -> valor a minimizar
def get_hvac_reward(obs):
    return NotImplementedError
