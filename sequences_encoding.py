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
import bezier

length = 500
nsample = 100
freq_avg = 1


#lambda seq

#blocky seq
def generate_blocky_seq(length, nblock_start, nblock_end):
    seq = []
    i = 0
    i_init = int(round(random.random()))
    seq.append(i_init)
    while i < length - 1:
        seq.append(seq[-1])
        


def generate_lambda_seq(length, autocorr_start, autocorr_end):
    seq = []
    seq.append(int(round(random.random())))
    for i in range(length - 1):
        if random.random() >= autocorr:
            seq.append(int(not seq[-1]))
        else:
            seq.append(seq[-1])
    return seq



def block_lengths(seq):
    blocks = []
    i = 1
    lblock = 0
    while i < len(seq):
        if i == len(seq) - 1:
            blocks.append(lblock)
        if seq[i] == seq[i-1]:
            lblock += 1
        else:
            blocks.append(lblock)
            lblock = 0
        i += 1
    return blocks

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
    #pdb.set_trace()
    mask = np.ma.array(diff / avg, mask = np.equal(avg, 0.))
    return np.ma.mean(mask)

def bezier_discrepancy(seq_1, seq_2):
    if len(seq_1) != len(seq_2):
        raise NameError("Sequences have different length")
    freq_avg = 1
    blocks_1 = block_lengths(seq_1)
    blocks_2 = block_lengths(seq_2)
    #hist_1 = np.histogram(blocks_1, bins = int(len(seq_1) / freq_avg),
    #                          range = [0.5 * freq_avg, len(seq_1) - 0.5 * freq_avg])
    #hist_2 = np.histogram(blocks_2, bins = int(len(seq_2) / freq_avg),
    #                          range = [0.5 * freq_avg, len(seq_2) - 0.5 * freq_avg])
    #node_x_1 = hist_1[1][1:] + hist_1[1][:-1]
    #node_x_2 = hist_2[1][1:] + hist_2[1][:-1]
    #nodes_1 = np.asfortranarray([node_x_1, hist_1[0]])
    #nodes_2 = np.asfortranarray([node_x_2, hist_2[0]])
    #curve_1 = bezier.curve(nodes_1, degree = 2)
    #curve_2 = bezier.curve(nodes_2, degree = 2)

discrepancies = {}
for autocorr in [0.5, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99]:
    discrepancies[autocorr] = {}
    for fft_length in [10, 40, 70, 100, 130, 160, 190]:
        discrepancy = []
        for s in range(nsample):
            seq = generate_seq(length, autocorr)
            restored_seq = truncate_fft(seq, fft_length)
        discrepancy.append(freq_discrepancy(seq, restored_seq, freq_avg))
        discrepancies[autocorr][fft_length] = sum(discrepancy) / len(discrepancy)
                           
data = pd.DataFrame.from_dict(discrepancies)

sns.set()
ax = sns.heatmap(data, annot = data, fmt='.1f')
plt.show()

print(data)
