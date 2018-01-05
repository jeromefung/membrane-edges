import numpy as np

def trim_data(lengths, tangents, condition, add2pi = True):
    output_lengths = lengths[condition]
    output_tangents = []
    for cond, tgt in zip(condition, tangents):
        if cond:
            temp_tgt = tgt.copy()
            if add2pi:
                temp_tgt[temp_tgt<0.]+= 2. * np.pi
            output_tangents.append(temp_tgt)
    return output_lengths, output_tangents

def trim_thickness(thickness, condition):
    thick_out = []
    len_out = []
    for thick, cond in zip(thickness, condition):
        if cond:
            thick_out.append(thick[thick.nonzero()])
            len_out.append(len(thick[thick.nonzero()]))
    return np.array(thick_out), np.array(len_out)
