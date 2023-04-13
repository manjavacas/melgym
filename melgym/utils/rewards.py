
def get_sgs_reward(obs, air_percent):
    '''
    Checks how many CVs have an air concentration lower to a given value.
    '''
    return sum(x for x in obs if x < air_percent)

# TO-DO: recorrer presiones y ver cuantas tienen una variacion < K
# Otra opcion: sumar las variaciones -> valor a minimizar
def get_pressure_compliance(obs):
    return NotImplementedError

