import tensorflow as tf
import numpy as np
import random

def sample_wavs(input_file, output_file, window_size, sample_len, sequential):
    input_wav = tf.io.read_file(input_file)
    output_wav = tf.io.read_file(output_file)

    input_samples, sample_rate = tf.audio.decode_wav(input_wav)
    output_samples, sample_rate = tf.audio.decode_wav(output_wav)
    
    min_index = window_size - 1 
    max_index = min(len(input_samples), len(output_samples))

    if sequential:
        max_index = max_index - sample_len

    if min_index > max_index:
        raise Exception("Input data too small to sample")

    input_data = np.zeros((sample_len, window_size, 1))
    output_data = np.zeros((sample_len, 1))

    start_index = random.randint(min_index, max_index)
    for i in range(sample_len):
        if sequential:
            sample_index = start_index + i
        else:
            sample_index = random.randint(min_index, max_index)
        input_data[i] = input_samples[sample_index - window_size : sample_index]
        output_data[i] = output_samples[sample_index]

    return input_data, output_data
    