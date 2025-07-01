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
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import Normalize
from shapely.geometry import Polygon

from label_prop import parameter_ordered_names, read_config
from label_prop import create_points, create_points2, fit_model, uncertainties, converge_points

import pdb

#FILTER = [0.2,0.2]
FILTER = [0., 0.]
#BP = { 'kernel' : 'rbf', 'gamma' : 20.,
#                  'max_iter' : 30, 'alpha' : 0.01}
BP = { 'kernel' : 'knn', 'n_neighbors' : 4,
                  'max_iter' : 30, 'alpha' : 0.01}
DEFAULT_PLOT_PARAMETERS = {
    'backend_params' : BP,
    'marker' : 'o',
    'label' : 'State probabilities',
    'filter' : [0., 0],
    'contour_linecolor' : 'black'}


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
    return np.array(histogram_edges)


def merge_adjacent_polygons(polygons, tolerance=0.0000001):
    """Merge polygons that share edges within tolerance"""
    merged = []
    visited = set()

    for i, poly1 in enumerate(polygons):
        if i in visited:
            continue
        current_union = poly1
        neighbors_found = True

        while neighbors_found:
            neighbors_found = False
            for j, poly2 in enumerate(polygons):
                if j not in visited and j != i and current_union.distance(poly2) < tolerance:
                    current_union = current_union.union(poly2)
                    visited.add(j)
                    neighbors_found = True

        merged.append(current_union)
        visited.add(i)

    return merged


def labels_to_distributions(labels):
    unique_labels = sorted(set(labels))
    n_unique_labels = len(unique_labels)
    label_distributions = []
    for label in labels:
        label_distribution = [0.] * n_unique_labels
        label_distribution[unique_labels.index(label)] = 1.
        label_distributions.append(label_distribution)
    return label_distributions


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description = 'Build a diagram of states plot')

    parser.add_argument('--config', metavar = 'JSON', type = str, nargs = '+',
        help = 'configuration file')

    args = parser.parse_args()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for config_filename in args.config:
        
        #pdb.set_trace()

        config = read_config(config_filename)

        mp, rp = config["model_parameters"], config["run_parameters"]
        pp = config.get("plot_parameters", DEFAULT_PLOT_PARAMETERS)

        p_names = parameter_ordered_names(mp)

        samples_df = pd.read_excel(rp["samples_file"])

        probed_points, probed_labels = create_points2(mp, samples_df,
                                                 return_unprobed = False)
        
        label_distributions = labels_to_distributions(probed_labels)
        
        converged_points, converged_distributions, _ = converge_points(probed_points, label_distributions)
        
        converged_labels = np.array([np.dot(i, [1,2]) for i in converged_distributions])
        
        #pdb.set_trace()
        
        
        he = histogram_edges(mp)
        
        boundary_points = [[he[0][0], he[1][0]], [he[0][0], he[1][1]],
                           [he[0][1], he[1][0]], [he[0][1], he[1][1]]]
        
        labels_with_boundary = np.concatenate((converged_labels, np.array([-1,-1,-1,-1])))
        
        points_with_boundary = np.vstack((converged_points, boundary_points))
        
        inverse_probed_points = [[point[1],point[0]] for point in points_with_boundary]
        
        scaler = MinMaxScaler()
        
        inverse_normalized_probed_points = scaler.fit_transform(inverse_probed_points)
        for point in inverse_probed_points:
            inverse_normalized_probed_points.append()

        points_labels = list(zip(probed_points, probed_labels))
        points1 = [pl[0] for pl in points_labels if int(pl[1]) == 1]
        points2 = [pl[0] for pl in points_labels if int(pl[1]) == 2]

        #pdb.set_trace()

        points1x = [p[0] for p in points1]
        points1y = [p[1] for p in points1]

        points2x = [p[0] for p in points2]
        points2y = [p[1] for p in points2]

        #ax = plt.subplot()

        #plt.scatter(points1y, points1x, s = 15, marker = pp['marker'], 
        #         facecolors = pp['colors']['1'], label = pp['label'])
        #plt.scatter(points2y, points2x, s = 15,
        #            marker = pp['marker'], facecolors = pp['colors']['2'])
        
        
        #vor = Voronoi(inverse_probed_points)
        vor = Voronoi(inverse_normalized_probed_points)
        
        norm = Normalize(vmin=min(probed_labels), vmax=max(probed_labels))
        cmap = plt.cm.viridis
        
        
        """
        for i, region in enumerate(vor.regions):
            if not -1 in region and len(region) > 0:
                polygon = [vor.vertices[j] for j in region]
                if labels_with_boundary[vor.point_region[i]] != -1:
                    ax.fill(*zip(*polygon), color=cmap(norm(labels_with_boundary[vor.point_region[i]])), alpha = 0.5)
        """
        label_1_polygons = []
        #for i in range(len(inverse_normalized_probed_points)):
        for i in range(len(inverse_probed_points)):
            if labels_with_boundary[i] <= 1.5:
                i_vor = vor.point_region[i]
                region = vor.regions[i_vor]
                if len(region) > 0 and -1 not in region:
                    polygon = [vor.vertices[j] for j in region]
                    label_1_polygons.append(Polygon(polygon))
                    #ax.fill(*zip(*polygon), facecolor = pp['voronoi_params']['facecolor'], alpha = 0.2)
        
        label_2_polygons = []
        #for i in range(len(inverse_normalized_probed_points)):
        for i in range(len(inverse_probed_points)):
            if labels_with_boundary[i] >= 1.5:
                i_vor = vor.point_region[i]
                region = vor.regions[i_vor]
                if len(region) > 0 and -1 not in region:
                    polygon = [vor.vertices[j] for j in region]
                    label_2_polygons.append(Polygon(polygon))
                
        merged_label_1_polygons = merge_adjacent_polygons(label_1_polygons)
        merged_label_2_polygons = merge_adjacent_polygons(label_2_polygons)
        
        pdb.set_trace()
        
        for polygon in label_1_polygons:
            #ax.fill(*zip(*polygon), facecolor = 'none', edgecolor = 'red', hatch = '/')
            if isinstance(polygon, Polygon):
                poly_x, poly_y = polygon.exterior.xy
                ax.fill(poly_x, poly_y, **pp["voronoi_params"])
            
        #for i in range(len(inverse_normalized_probed_points)):
        for i in range(len(inverse_probed_points)):
            region = vor.regions[i_vor]
            if -1 not in region and len(region) > 0:
                polygon = [vor.vertices[j] for j in region]
                if labels_with_boundary[i] == 1:
                    pass
                    #ax.fill(*zip(*polygon), color=cmap(norm(labels_with_boundary[i])), alpha = 0.5)
                    #ax.fill(*zip(*polygon), pp['voronoi_params']['facecolor'])
        
        """
        #voronoi_plot_2d(vor, ax = ax, show_points=True, show_vertices=False, line_width=1)
        
        
        
        #scatter = ax.scatter(probed_points[:, 0], probed_points[:, 1], c=probed_labels, cmap=cmap, 
        #            edgecolors='black', s=2)


        
        
            
        grid_points_v1, grid_labels_v1 = create_points2(mp, samples_df,
                                             return_unprobed = True,
                                             unprobed_backend = "v1")
        
        grid_points_v2, grid_labels_v2 = create_points2(mp, samples_df,
                                             return_unprobed = True,
                                             unprobed_backend = "v2")
        
        if len(grid_points_v2) != len(grid_points_v1):
            grid_points = grid_points_v1
            grid_labels = grid_labels_v1
            #raise NameError(f"v1: {len(grid_points_v1)} points, v2 {len(grid_points_v2)} points")
        else:
            grid_points = grid_points_v1
            grid_labels = grid_labels_v1

        uc_features, points, distributions = fit_model(mp,
                                                   grid_points, grid_labels,
                                                   mode = rp["sampling_mode"],
                                        backend_params = pp['backend_params'])

        param1_values = [i[0] for i in points]
        param2_values = [i[1] for i in points]
        rel_prob_values = [i[0]/(i[0]+i[1]) for i in distributions]

        uncertainties_list = uncertainties(distributions, mode = "EB")

        heatmap, param1_edges, param2_edges = np.histogram2d(param1_values,
                                                         param2_values,
                                            bins = histogram_nbins(mp),
                                            range = histogram_edges(mp),
                                            weights = rel_prob_values)

        uc_heatmap, _, _ = np.histogram2d(param1_values,
                                          param2_values,
                                          bins = histogram_nbins(mp),
                                          range = histogram_edges(mp),
                                          weights = uncertainties_list)

        max_uncertainty = np.max(uncertainties_list)

        param1_centers = (param1_edges[:-1] + param1_edges[1:])/2
        param2_centers = (param2_edges[:-1] + param2_edges[1:])/2
        smoothed_heatmap = scipy.ndimage.filters.gaussian_filter(heatmap, pp['filter'])

        plt.contour(param2_centers, param1_centers, smoothed_heatmap,
                    [0.5], linewidths = pp['contour_linewidth'],
                            colors = pp['contour_linecolor'])

        #plt.contour(param2_centers, param1_centers, uc_heatmap,
        #            [0.9*max_uncertainty], linestyles = 'dashed',
        #                linewidth = 2, colors = "black")
        """
        

    plt.xlabel('$N_{cut}$')
    plt.ylabel('$f_{mod}$, %')
    plt.xlim((0.,16))
    plt.ylim((0,1))
    plt.rcParams["figure.figsize"] = (8,6)

    plt.show()