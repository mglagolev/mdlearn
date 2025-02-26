#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import random
import os
import argparse
import json
import itertools
from diagram_viz import plot_uncertainties, plot_distributions

# Function definitions

def read_config(config_filename):

    with open(config_filename, "r") as f:
         config = json.load(f)
         
    try:
        points_file = config["points_file"]
        config["run_parameters"]["p_mode"] = "file"
        model_parameters, _ = read_points(points_file)
        config["model_parameters"] = model_parameters
    except KeyError:
        try:
            model_parameters = config["model_parameters"]
            config["run_parameters"]["p_mode"] = "grid"
            points_file = None
        except KeyError:
            raise NameError("Either points_file or model_parameters sections\
 should be present in the configuration file")

    return config


def parameter_ordered_names(parameters):
    if type(parameters) == dict:
        p_names = list(parameters.keys())
    elif type(parameters) == list:
        p_names = parameters
    else:
        raise NameError(f"Unsupported parameters type {type(parameters)}")
    p_names.sort()
    return p_names


def read_points(points_file):
    if points_file[-4:] == ".xls" or points_file[-5:] == ".xlsx":
        points_df = pd.read_excel(points_file)
    elif points_file[-4:] == ".csv":
        points_df = pd.read_csv(points_file)

    p_names = parameter_ordered_names(list(points_df.columns))
    parameters = {}
    points_list = []
    for p_name in p_names:
        parameter_values = points_df[p_name]
        parameter_min = min(parameter_values)
        parameter_max = max(parameter_values)
        parameters[p_name] = {"min" : parameter_min, "max" : parameter_max}
        points_list.append(parameter_values)
    points_array = np.array(points_list)
    return parameters, points_array.T


def normalize(parameter, value):
    if parameter["max"] != parameter["min"]:
        return (value - parameter["min"]) / (parameter["max"] - parameter["min"])
    else:
        return value


def restore(parameter, value):
    if parameter["max"] != parameter["min"]:
        return parameter["min"] + value * (parameter["max"] - parameter["min"])
    else:
        return value


#def nvalues(parameter):
#    return int((parameter["max"] - parameter["min"]) / parameter["step"]) + 1


def create_grid(parameters, mode = 'itertools'):
    p_names = parameter_ordered_names(parameters)
    if mode == 'mgrid':
        """slices = []
        for p_name in p_names:
            slices.append(slice(parameters[p_name]["min"],
                          parameters[p_name]["max"]+parameters[p_name]["step"],
                          parameters[p_name]["step"]))
        points = np.mgrid[*slices].reshape(len(slices),-1).T"""
        raise NameError("Not implemented in older versions of Python")
    elif mode == 'itertools':
        p_ranges = []
        for p_name in p_names:
            p_range = np.arange(parameters[p_name]["min"],
                                parameters[p_name]["max"]
                              + parameters[p_name]["step"],
                                parameters[p_name]["step"])
            if p_range[-1] > parameters[p_name]["max"]:
                p_range = p_range[:-1]
            p_ranges.append(p_range)
        points = list(itertools.product(*p_ranges))
    return points


def create_points(parameters, samples_df, p_mode = "grid", points_file = None,
                  return_unprobed = True):
    p_names = parameter_ordered_names(parameters)
    if p_mode == "grid":
        points_to_check = create_grid(parameters)
    elif p_mode == "file":
        _, points_to_check = read_points(points_file)
    else:
        raise NameError(f"Unsupported point mode {p_mode}")
    points = []
    labels = []
    for point in points_to_check:
        query_array = []
        for i_p_name in range(len(p_names)):
            query_array.append(f"{p_names[i_p_name]} == {point[i_p_name]}")
        query = " & ".join(query_array)
        samples = samples_df.query(query)
        if len(samples) > 0:
            points.append(point)
            labels.append(samples["state"].iloc[0])
        elif return_unprobed:
            points.append(point)
            labels.append(-1)
    return points, labels


def normalize_points(parameters, points):
    p_names = parameter_ordered_names(parameters)
    normalized_points = []
    for point in points:
        normalized_point = []
        for i_p_name in range(len(p_names)):
            normalized_point.append(normalize(parameters[p_names[i_p_name]],
                                              point[i_p_name]))
        normalized_points.append(normalized_point)
    return normalized_points


def restore_points(parameters, points):
    p_names = parameter_ordered_names(parameters)
    restored_points = []
    for point in points:
        restored_point = []
        for i_p_name in range(len(p_names)):
            restored_point.append(restore(parameters[p_names[i_p_name]],
                                          point[i_p_name]))
        restored_points.append(restored_point)
    return restored_points


def uncertainties_eb(label_distributions):
    """ Entropy-based: sum(p*log(p)) """
    uncertainties = []
    for sample in label_distributions:
        if np.max(sample) == 1.:
            uncertainties.append(0.)
        else:
            uncertainties.append(-1. * np.dot(sample, np.log(sample)))
    return uncertainties


def uncertainties_lc(label_distributions):
    """ Least certain: 1 - max(X,p) """
    uncertainties = []
    for sample in label_distributions:
        uncertainties.append(1. - np.max(sample))
    return uncertainties


def uncertainties_ms(label_distributions):
    """ Margin sampling: 1 - [max(X,p) - second_max(X,p)] """
    uncertainties = []
    for sample in label_distributions:
        prob_ix = np.argsort(sample)
        uncertainties.append(1. - sample[prob_ix[-1]] + sample[prob_ix[-2]])
    return uncertainties


def uncertainties(label_distributions, mode = None):
    match mode:
        case "EB":
            return uncertainties_eb(label_distributions)
        case "LC":
            return uncertainties_lc(label_distributions)
        case "MS":
            return uncertainties_ms(label_distributions)


def fit_model(parameters, points, labels, mode = None):

    normalized_points = normalize_points(parameters, points)

    label_prop = LabelSpreading(kernel = 'rbf', gamma = 20,
                                max_iter = 100000)

    label_prop.fit(normalized_points, labels)

    distributions = label_prop.label_distributions_

    uc = uncertainties(distributions, mode = mode)

    uc_features = list(zip(uc, points))

    uc_features.sort(key = lambda k: k[0])
    
    return uc_features, points, distributions


def choose_random_unprobed(points, labels):
    points_labels = zip(points, labels)
    unprobed_points = [p_l[0] for p_l in points_labels if p_l[1] == -1]
    return random.choice(unprobed_points)


def choose_parameters(config, samples_df):

    mode = config["run_parameters"]["sampling_mode"]
    uc_threshold = config["run_parameters"]["uc_threshold"]
    p_mode = config["run_parameters"]["p_mode"]
    parameters = config["model_parameters"]
    
    if p_mode == "points":
        points_file = config["points_file"]
    
    points, labels = create_points(parameters, samples_df, p_mode = p_mode,
                                   points_file = points_file)
    
    if samples_df.shape[0] > config["run_parameters"]["n_random"]:
    
        uc_features, _, _ = fit_model(parameters, points, labels, mode = mode)

        uc_delta = uc_features[-1][0] - uc_features[0][0]

        if uc_delta > uc_threshold:
            return uc_features[-1][1], mode
        else:
            mode = "random"
            return choose_random_unprobed(points,labels), mode
            #return uc_features[random.randrange(0,len(uc_features))][1]
    else:
        mode = "random"
        return choose_random_unprobed(points, labels), mode


def label(simulation, u):
    r_neigh_aggregation = simulation["run_parameters"]["r_neigh_aggregation"]
    aggregates_dict = determine_aggregates(u, r_neigh = r_neigh_aggregation)
    aggregates_list = aggregates_dict["data"][list(aggregates_dict["data"].keys())[0]]
    if len(aggregates_list) == 1:
        return 1
    elif len(aggregates_list) > 1:
        return 2
    else:
        raise NameError(f"Aggregates list length is {len(aggregates_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Build a phase diagram')

    #parser.add_argument('--samples', metavar = 'XLS', type = str, nargs = 1,
    #    help = 'file with datapoints')

    parser.add_argument('--config', metavar = 'JSON', type = str, nargs = 1,
        help = 'configuration file')


    args = parser.parse_args()

    config_filename = args.config[0]

    config = read_config(config_filename)
    
    mp, rp = config["model_parameters"], config["run_parameters"]
    
    p_names = parameter_ordered_names(mp)

    samples_df = pd.read_excel(rp["samples_input"])
    
    """if args.plot:
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap
        import seaborn as sns
        points, labels = create_points(model_parameters, samples_df)
        uc_features, points, distributions = fit_model(model_parameters,
                                                       points, labels,
                                                       mode = rp["sampling_mode"])
        plot_uncertainties(model_parameters, uc_features)
        plot_distributions(model_parameters, points, distributions, samples_file = samples_in_filename)
"""

    if True:
        import MDAnalysis as mda
        try:
            from mouse2.mouse2.lib.aggregation import determine_aggregates
        except ModuleNotFoundError:
            from mouse2.lib.aggregation import determine_aggregates
        #from microplastic.modify_seq import modify_seq
        from microplastic.modify_seq import prepare_sample
        from parzen_search import substitute_values
        # Main loop
        for i_iter in range(samples_df.shape[0] + 1, rp["iterations"] + 1):
            # Fit the model and choose simulation parameters
            run_parameters, choice_mode = choose_parameters(config, samples_df)
            # Dump the model data

            # Create the simulation dict object with all the attributes stored
            simulation = {}
            simulation["sample"] = i_iter
            simulation["choice"] = choice_mode
            simulation["config"] = config
            for i_param in range(len(p_names)):
                simulation[p_names[i_param]] = run_parameters[i_param]
            #simulation[p_names[0]] = run_parameters[0]
            #simulation[p_names[1]] = run_parameters[1]
            # Run the simulation
            for i_step in range(1, rp["n_steps"] + 1):
                run_filename = f"run_{i_iter}.{i_step}.lammps"
                infile_name = f"in_{i_iter}.{i_step}.data"
                prev_outfile_name = f"out_{i_iter}.{i_step-1}.data"
                outfile_name = f"out_{i_iter}.{i_step}.data"
                logfile_name = f"{i_iter}.{i_step}.log"
                xdata_name = f"xdata.{i_iter}.{i_step}.lammps"
                dump_name = f"atoms.{i_iter}.{i_step}.lammpsdump"
                substitute_values(rp["run_template"], run_filename,
                                  [["INPUT", infile_name],
                                   ["OUTPUT", outfile_name],
                                   ["LOG", logfile_name],
                                   ["XDATA", xdata_name],
                                   ["DUMP", dump_name]
                                   ])
                if i_step == 1:
                    #u = mda.Universe(initial_sequence_file)
                    u = prepare_sample(simulation)
                    #modify_seq(u, prob = simulation["f"],
                    #           nsplit = simulation["nsplit"])
                    u.atoms.write(infile_name)
                    simulation["step"] = 1
                else:
                    os.system(f"cp -a {prev_outfile_name} {infile_name}")
                    simulation["step"] += 1
                if rp["run_mode"] == "module":
                    from lammps import lammps
                    lmp = lammps()
                    lmp.file(run_filename)
                elif rp["run_mode"] == "standalone":
                    run_options = rp["run_options"]
                    command = "/mnt/share/glagolev/run_online.py " \
                            + f"--input {run_filename} {run_options}"
                    exit_code = os.system(command)
            # Process the simulation data: determine the aggregation number
                output_exists = os.path.isfile(outfile_name)
                actions = eval(rp["actions"])
                if output_exists:
                    u = mda.Universe(outfile_name)
                    simulation["state"] = label(simulation, u)
                else:
                    simulation["state"] = 0
                if actions[simulation["state"]] == "break":
                    break
            # Update the dataframe
            simulation.pop("config")
            new_df = pd.DataFrame(simulation, index = [0])
            updated_df = pd.concat([samples_df, new_df], ignore_index = True)
            updated_df.reset_index()
            updated_df.to_excel(rp["samples_output"])
            samples_df = updated_df