import numpy as np
from melkit.toolkit import Toolkit


def make_sgs_observation(input_file, edf_file):
    '''
    Returns an observation from MELCOR EDF output.
    Each observation is a vector [x_1, x_2, ... x_N] with x_i being the 
    final air concentration percentage for each Ar-filled CV.
    '''
    toolkit = Toolkit(input_file)

    edf_data = toolkit.as_dataframe(edf_file).iloc[-1, 1:]

    return np.array(edf_data, dtype=np.float32)


def make_observation(input_file, edf_file):
    '''
    Returns an observation from MELCOR EDF output.
    Each observation is a vector [x_1, x_2..., x_N] with x_i being the mean deviation
    of each CVH.P with respect to its initial value.
    '''

    toolkit = Toolkit(input_file)
    edf_data = toolkit.as_dataframe(edf_file).iloc[:, 1:]

    observation = []

    for col_name in edf_data.columns:

        # Get CV ID
        dot_pos = col_name.find('.')
        cv_num = int(col_name[dot_pos+1:])
        cv_id = 'CV'
        if cv_num < 10:
            cv_id += str(cv_num).zfill(3)
        elif cv_num < 100:
            cv_id += str(cv_num).zfill(2)
        else:
            cv_id += str(cv_num)

        # Get initial pressure
        cv = toolkit.get_cv(cv_id)
        cv_pressure = float(cv.get_field('PVOL'))

        # Compute mean deviation
        deviation = edf_data[col_name].sub(cv_pressure).abs().mean()
        observation.append(deviation)

    return observation
