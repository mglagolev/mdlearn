#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:23:31 2025

@author: misha
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import scipy

from label_prop import parameter_ordered_names, read_config
from label_prop import create_points, create_points2, fit_model, uncertainties

import pdb

FILTER = [6,4]

def nvalues(parameter):
    return int((parameter["max"] - parameter["min"]) / parameter["step"]) + 1

def histogram_nbins(parameters):
    p_names = parameter_ordered_names(parameters)
    histogram_nbins = []
    for p_name in p_names:
        parameter = parameters[p_name]
        histogram_nbins.append(nvalues(parameter))
    return histogram_nbins

def histogram_edges(parameters):
    p_names = parameter_ordered_names(parameters)
    histogram_edges = []
    for p_name in p_names:
        parameter = parameters[p_name]
        histogram_min = parameter["min"] - parameter["step"] / 2
        histogram_max = parameter["max"] + parameter["step"] / 2
        histogram_edges.append([histogram_min, histogram_max])
    return histogram_edges

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description = 'Build a diagram of states plot')

    parser.add_argument('--config', metavar = 'JSON', type = str, nargs = 1,
        help = 'configuration file')

    args = parser.parse_args()

    config_filename = args.config[0]

    config = read_config(config_filename)

    mp, rp = config["model_parameters"], config["run_parameters"]

    p_names = parameter_ordered_names(mp)

    samples_df = pd.read_excel(rp["samples_output"])

    probed_points, probed_labels = create_points2(mp, samples_df,
                                                 return_unprobed = False)

    points_labels = list(zip(probed_points, probed_labels))
    points1 = [pl[0] for pl in points_labels if int(pl[1]) == 1]
    points2 = [pl[0] for pl in points_labels if int(pl[1]) == 2]
    
    #pdb.set_trace()
    
    points1x = [p[0] for p in points1]
    points1y = [p[1] for p in points1]
    
    points2x = [p[0] for p in points2]
    points2y = [p[1] for p in points2]

    #ax = plt.subplot()

    plt.scatter(points1y, points1x, s = 15, marker = 'o', 
                 facecolors = 'red')
    plt.scatter(points2y, points2x, s = 15, marker = 'o', facecolors = 'blue')

    grid_points, grid_labels = create_points2(mp, samples_df,
                                             return_unprobed = True)

    uc_features, points, distributions = fit_model(mp,
                                                   grid_points, grid_labels,
                                                   mode = rp["sampling_mode"])

    param1_values = [i[0] for i in points]
    param2_values = [i[1] for i in points]
    rel_prob_values = [i[0]/(i[0]+i[1]) for i in distributions]
    
    #pdb.set_trace()
    
    uncertainties = uncertainties(distributions, mode = "EB")

    heatmap, param1_edges, param2_edges = np.histogram2d(param1_values,
                                                         param2_values,
                                            bins = histogram_nbins(mp),
                                            range = histogram_edges(mp),
                                            weights = rel_prob_values)

    uc_heatmap, _, _ = np.histogram2d(param1_values,
                                      param2_values,
                                      bins = histogram_nbins(mp),
                                      range = histogram_edges(mp),
                                      weights = uncertainties)

    param1_centers = (param1_edges[:-1] + param1_edges[1:])/2
    param2_centers = (param2_edges[:-1] + param2_edges[1:])/2
    smoothed_heatmap = scipy.ndimage.filters.gaussian_filter(heatmap,FILTER)

    plt.contour(param2_centers, param1_centers, smoothed_heatmap,
                [0.5], linewidth = 2, colors = "black")

    #plt.contour(param2_centers, param1_centers, uc_heatmap,
    #            [0.3], linestyles = 'dashed', linewidth = 2, colors = "black")

    plt.xlabel('$n_{cut}$')
    plt.ylabel('$f_{mod}$, %')
    plt.rcParams["figure.figsize"] = (8,6)

    plt.show()