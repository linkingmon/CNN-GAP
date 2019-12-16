import numpy as np


def pad_sequences(data_list, maxlen):
    new_data_list = []
    for waveform in data_list:
        if len(waveform) < maxlen:
            remainder = maxlen - len(waveform)
            new_data_list.append(np.pad(waveform, (int(remainder/2), remainder - int(remainder/2)), 'constant', constant_values=0))
        else:
            new_data_list.append(waveform[:maxlen])
    return np.array(new_data_list)
