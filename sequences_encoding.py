#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import signal


length = 200
nsample = 100000
choices = [0, 1]
freq_avg = 1
maxblock = 1

def signal_handler(signum, frame):
    raise Exception("Timed out!")

def generate_blocked_markov(length, p_matrix, block_length_pairs,
                            alternating = False, glued = True, choices = None):
    """
    Generate a Markov chain from a set of choices.
    Depending on block lengths, the sequences can be formed not from individual
    elements, but of blocks of the specified length. The length of a block can
    be varied along the sequence by interpolating between the initial and the
    final value.

    Parameters
    ----------
    length : TYPE
        DESCRIPTION.
    p_matrix: np.ndarray((n_types, n_types), dtype = float)
        Markov probability matrix
    block_length_pairs : np.ndarray((n_types, 2), dtype = int)
        [[Astart, Aend],
         [Bstart, Bend],
         [Cstart, Cend],
         ...]

    Returns
    -------
    sequence of choices

    """
    n_types = len(block_length_pairs)
    p_matrix = np.array(p_matrix)
    if p_matrix.shape != (n_types, n_types):
        raise NameError("""The dimensions of the probability matrix should be
    equal to the length of the block length array of tuples""")
    if choices == None:
        choices = np.arange(0, n_types, dtype = int)
    elif len(choices) != n_types:
        raise NameError("""Length of the sequence values must be equal to the
                        outer dimension of the block length matrix""")
    i_choices = np.arange(0, n_types, dtype = int)
    seq = []
    i = 0
    block_type = random.randint(0, len(choices) - 1) # integer
    while i < length:
        if not alternating:
            if glued:
                block_type = random.choices(i_choices, k = 1,
                                            weights = p_matrix[block_type])[0]
            else:
                current_i_choices = np.concatenate([i_choices[:block_type],
                                                i_choices[block_type + 1:]])
                current_p_list = np.concatenate(
                    [p_matrix[block_type][:block_type],
                    p_matrix[block_type][block_type + 1:]])
                block_type = random.choices(current_i_choices, k = 1,
                                           weights = current_p_list)[0]
        else:
            block_type = ( block_type + 1 ) % n_types
        block_lengths = block_length_pairs[block_type]
        position_vector = [1. - i / (length - 1.), i / (length - 1.)]
        block = int( (block_lengths[0] * position_vector[0]
                    + block_lengths[1] * position_vector[1])
                / (1. + (block_lengths[0] - block_lengths[1])/ 2. / length))
        #block = int(np.dot(block_lengths, position_vector)
        #         / (1. + (block_lengths[0] - block_lengths[1])/ 2. / length))
        for j in range(block):
            seq.append(choices[block_type])
            i += 1
            if i >= length:
                print("in_block_cut")
                return seq
    return seq


def transition_probability_matrix(sequence):
    """Calculate the actual transition probability matrix of a sequence"""
    types = list(set(sequence))
    types.sort()
    n_types = len(types)
    matrix = np.zeros((n_types, n_types), dtype = float)
    for k in range(len(sequence) - 1):
        i_index = types.index(sequence[k])
        j_index = types.index(sequence[k+1])
        matrix[i_index][j_index] += 1
    for i in range(matrix.shape[0]):
        matrix[i] = matrix[i] / np.sum(matrix[i])
    return matrix, types


def block_lengths(sequence):
    """Return the actual block lengths in a sequence"""
    types = list(set(sequence))
    types.sort()
    blocks = {}
    for block_type in types:
        blocks[block_type] = []
    i = 1
    block_type = sequence[0]
    block_length = 1
    while i <= len(sequence):
        if i == len(sequence):
            blocks[block_type].append(block_length)
            break
        if sequence[i] == sequence[i-1]:
            block_length += 1
        else:
            blocks[block_type].append(block_length)
            block_type = sequence[i]
            block_length = 1
        i += 1
    return blocks


def block_length_distributions(sequence):
    """Return the histogram of block lengths in a sequence"""
    blocks = block_lengths(sequence)
    frequencies = {}
    for block_type in blocks:
        type_frequencies, bin_edges = np.histogram(
                     blocks[block_type], bins = int(len(sequence) / freq_avg),
                     range = [0, len(sequence)])
        frequencies[block_type] = type_frequencies
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    return frequencies, bin_centers


def check_generator(length = None, p_matrix = None, block_length_pairs = None,
                            alternating = None, glued = None):
    """Generate a sequence, then plot the source transition matrix and block
    length parameters, and the actual transition matrix and block length
    distributions"""
    if length == None:
        length = random.randint(50, 1000)
    if p_matrix == None:
        n_types = random.randint(2, 6)
        p_matrix = np.ndarray((n_types, n_types), dtype = float)
        for i in range(p_matrix.shape[0]):
            for j in range(p_matrix.shape[1]):
                p_matrix[i][j] = random.uniform(0., 1.)
            p_matrix[i] = p_matrix[i] / np.sum(p_matrix[i])
    elif block_length_pairs == None:
        p_matrix = np.array(p_matrix)
        n_types = p_matrix.shape[0]
    if block_length_pairs == None:
        block_length_pairs = np.ndarray((n_types, 2), dtype = int)
        for i in range(n_types):
            for j in range(2):
                block_length_pairs[i][j] = random.randint(1, maxblock)
    if alternating == None:
        alternating = random.choice([True, False])
    if glued == None:
        glued = random.choice([True, False])

    sequence = generate_blocked_markov(length, p_matrix, block_length_pairs,
                    alternating = alternating, glued = glued, choices = None)
    pd_tgt_p = pd.DataFrame(p_matrix,
                              columns = np.arange(0, n_types, dtype = int),
                              index = np.arange(0, n_types, dtype = int))
    actual_p_matrix, actual_types = transition_probability_matrix(sequence)
    print(p_matrix)
    print(actual_p_matrix)
    pd_act_p = pd.DataFrame(actual_p_matrix,
                              columns = actual_types,
                              index = actual_types)
    frequencies, block_lengths = block_length_distributions(sequence)
    frequencies_dict = {}
    cutoff = 0
    for block_type in frequencies:
        cutoff = max(cutoff, np.max(np.where(frequencies[block_type])))
    for block_type in frequencies:
        frequencies_dict[block_type] = {}
        for i in range(cutoff):
            frequencies_dict[block_type][block_lengths[i]]\
                = frequencies[block_type][i]
    #figure1 = plt.figure(1)
    #fig, axs = plt.subplots(ncols=2)
    plt.subplot(111)
    sns.heatmap(pd_tgt_p, annot = pd_tgt_p, fmt='.2f') #, ax = axs[0])
    plt.subplot(212)
    sns.heatmap(pd_act_p, annot = pd_act_p, fmt='.2f') #, ax = axs[1])
    #plt.show()
    #figure1.show()
    #figure2 = plt.figure(2)
    plt.subplot(122)
    plt.xlim([1,cutoff])
    for block_type in frequencies:
        plt.plot(block_lengths, frequencies[block_type], label=str(block_type))
    plt.show()
    #pd_block_distributions = pd.DataFrame.from_dict(frequencies_dict)
    #sns.lineplot(pd_block_distributions)
    #plt.show()
    #for blocktype in frequencies:
    #    plt.plot(frequencies[block_type], block_lengths, ax = axs[2])
    
    

def truncate_fft(seq, fft_length):
    seq_fft = sp.fft.fft(seq)
    l_truncated = list(seq_fft[:fft_length]) + [0.] * (len(seq_fft) - fft_length)
    truncated = np.array(l_truncated)
    return np.flip(np.round(np.absolute(sp.fft.fft(truncated) / len(truncated))))


def seq_discrepancy(seq_1, seq_2):
    if len(seq_1) != len(seq_2):
        raise NameError("Sequences have different length")
    discrepancy = 0.
    for i in range(len(seq_1)):
        if seq_1[i] != seq_2[i]:
            discrepancy += 1
    discrepancy /= len(seq_1)
    return discrepancy


def freq_discrepancy(seq_1, seq_2, freq_avg):
    if len(seq_1) != len(seq_2):
        raise NameError("Sequences have different length")
    blocks_1 = block_lengths(seq_1)
    blocks_2 = block_lengths(seq_2)
    freq_1 = np.histogram(blocks_1, bins = int(len(seq_1) / freq_avg),
                    range = [0.5 * freq_avg, len(seq_1) - 0.5 * freq_avg])[0]
    freq_2 = np.histogram(blocks_2, bins = int(len(seq_2) / freq_avg),
                    range = [0.5 * freq_avg, len(seq_2) - 0.5 * freq_avg])[0]
    diff = np.abs(freq_1 - freq_2)
    avg = 0.5 * (freq_1 + freq_2)
    mask = np.ma.array(diff / avg, mask = np.equal(avg, 0.))
    return np.ma.mean(mask)


if __name__ == "__main__":
    """Generate a number of sequences using optuna trial"""
    
    import optuna
    
    def objective(trial: optuna.trial.Trial):
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alararm(2)   # Two seconds
        try:
            block_0_start = trial.suggest_int("block_0_start", 0, int(length / 2))
            block_0_end = trial.suggest_int("block_0_end", 0, int(length / 2))
            block_1_start = trial.suggest_int("block_1_start", 0, int(length / 2))
            block_1_end = trial.suggest_int("block_1_end", 0, int(length / 2))
        
            pii = trial.suggest_float("pii", 0., 1.)
            pij = trial.suggest_float("pij", 0., 1.)
            pji = trial.suggest_float("pji", 0., 1.)
            pjj = trial.suggest_float("pjj", 0., 1.)
        
            alternating = trial.suggest_categorical("alternating", [True, False])
            if len(choices <= 2):
                glued = True
            else:
                glued = trial.suggest_categorical("glued", [True, False])
        
            p_matrix = [[pii, pij], [pji, pjj]]
            block_length_pairs = [[block_0_start, block_0_end],
                              [block_1_start, block_1_end]]
        
            seq = generate_blocked_markov(length, p_matrix, block_length_pairs,
                                    alternating = alternating, glued = glued,
                                    choices = choices)
            sequences.append(seq)
            return 0
        except:
            return 1
    
    sequences = []
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(direction="minimize", sampler = sampler)
    study.optimize(objective, n_trials=nsample)

    sequences = np.array(sequences)
    np.save(f'sequences_N{length}_n{nsample}.npy', sequences)
