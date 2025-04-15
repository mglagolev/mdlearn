#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:21:48 2024

@author: misha
"""
import argparse
import json
from simple_pid import PID
import os
import pandas as pd
import numpy as np
from mouse2.tests.create_configuration import create_configuration


class NCalls():
    def __init__(self):
        self.counter = 0
    def call(self):
        self.counter += 1
        return self.counter


def substitute_values(template_filename, output_filename,
                      substitutions = None):
    template_file = open(template_filename, 'r')
    template = template_file.read()
    template_file.close()
    if type(substitutions) == list:
        for substitution in substitutions:
            template = template.replace(*substitution)
    elif type(substitutions) == dict:
        for key in substitutions:
            template = template.replace(key, substitutions[key])
    run_file = open(output_filename, 'w')
    run_file.write(template)
    run_file.close()


def feedback_reader(parameter_name, args):
    if parameter_name == "e_pair":
        feedback = e_pair_reader(**args)
    elif parameter_name == "diff_e_pair":
        feedback = diff_e_pair_reader(**args)
    elif parameter_name == "c_ratio":
        feedback = c_ratio_reader(**args)
    else:
        raise NameError(f"Feedback reader for {parameter_name} not implemented")
    return feedback


def e_pair_reader(file = None):
    # Read epair from file
    df = pd.read_csv(file, delimiter = ' ', skiprows = 2, header = None)
    data_length = df.shape[0]
    e_pair_data = df[1][int(data_length/2):]
    return e_pair_data.mean()


def diff_e_pair_reader(file1 = None, file2 = None, factor1 = None, factor2 = None):
    # Read epair from file
    df1 = pd.read_csv(file1, delimiter = ' ', skiprows = 2, header = None)
    data1_length = df1.shape[0]
    e_pair1_data = df1[1][int(data1_length/2):]
    e_pair1_mean = e_pair1_data.mean()
    
    df2 = pd.read_csv(file2, delimiter = ' ', skiprows = 2, header = None)
    data2_length = df2.shape[0]
    e_pair2_data = df2[1][int(data2_length/2):]
    e_pair2_mean = e_pair2_data.mean()
    return factor1 * e_pair1_mean + factor2 * e_pair2_mean


def c_ratio_reader(data = None, atomdump = None):
    import MDAnalysis as mda
    from microplastic.c_ratio import c_ratio
    u = mda.Universe(data, atomdump)
    c_ratio, err = c_ratio(u)
    return c_ratio


def prepare_simulation(simulation):
    # Prepare the simulation
    config = simulation["config"]
    i_iter = simulation["iteration"]
    in_filename = f"{config['run_parameters']['in_filename_template']}{i_iter}.data"
    out_filename = f"{config['run_parameters']['out_filename_template']}{i_iter}.data"
    run_filename = f"{config['run_parameters']['run_filename_template']}{i_iter}.lammps"
    log_filename = f"{config['run_parameters']['log_filename_template']}{i_iter}.lammps"
    template = config["run_parameters"]["template_filename"]
    script_changes = []
    script_changes.append(["INPUT", in_filename])
    script_changes.append(["OUTPUT", out_filename])
    script_changes.append(["LOG", log_filename])
    print(f"Step {i_iter}")
    simulation["controls"] = {}
    pids = simulation["pids"]
    feedbacks = simulation["feedbacks"]
    timer = simulation["timer"]
    for param, details in config["model_parameters"].items():
        if i_iter == 0:
            p = details["p"]
            i = details["i"]
            d = details["d"]
            setpoint = details["setpoint"]
            output_limits = eval(details["limits"])
            pids[param] = PID(p, i, d, setpoint = setpoint,
                                      output_limits = output_limits,
                                      time_fn = timer.call)
            control = details["initial"] + details["shift"]
            print(f"Control {param} is {control}")
        else:
            feedback = feedbacks[details["output"]]
            control = pids[param](feedback) + details["shift"]
            print(f"Control {param} is {control}")
        script_changes.append([param, str(control)])
        simulation["controls"][param] = control

    substitute_values(template,run_filename,script_changes)


def prepare_differential_simulation(simulation):
    # Prepare the simulation
    config = simulation["config"]
    i_iter = simulation["iteration"]
    in1_filename = f"{config['run_parameters']['in1_filename_template']}{i_iter}.data"
    out1_filename = f"{config['run_parameters']['out1_filename_template']}{i_iter}.data"
    run1_filename = f"{config['run_parameters']['run1_filename_template']}{i_iter}.lammps"
    log1_filename = f"{config['run_parameters']['log1_filename_template']}{i_iter}.lammps"
    
    in2_filename = f"{config['run_parameters']['in2_filename_template']}{i_iter}.data"
    out2_filename = f"{config['run_parameters']['out2_filename_template']}{i_iter}.data"
    run2_filename = f"{config['run_parameters']['run2_filename_template']}{i_iter}.lammps"
    log2_filename = f"{config['run_parameters']['log2_filename_template']}{i_iter}.lammps"
    
    template1 = config["run_parameters"]["template1_filename"]
    template2 = config["run_parameters"]["template2_filename"]
    
    script1_changes = []
    script1_changes.append(["INPUT", in1_filename])
    script1_changes.append(["OUTPUT", out1_filename])
    script1_changes.append(["LOG", log1_filename])
    
    script2_changes = []
    script2_changes.append(["INPUT", in2_filename])
    script2_changes.append(["OUTPUT", out2_filename])
    script2_changes.append(["LOG", log2_filename])
    
    print(f"Step {i_iter}")
    simulation["controls"] = {}
    pids = simulation["pids"]
    feedbacks = simulation["feedbacks"]
    timer = simulation["timer"]
    for param, details in config["model_parameters"].items():
        if i_iter == 0:
            p = details["p"]
            i = details["i"]
            d = details["d"]
            setpoint = details["setpoint"]
            output_limits = eval(details["limits"])
            pids[param] = PID(p, i, d, setpoint = setpoint,
                                      output_limits = output_limits,
                                      time_fn = timer.call)
            control = details["initial"] + details["shift"]
            print(f"Control {param} is {control}")
        else:
            feedback = feedbacks[details["output"]]
            control = pids[param](feedback) + details["shift"]
            print(f"Control {param} is {control}")
        script1_changes.append([param, str(control)])
        script2_changes.append([param, str(control)])
        simulation["controls"][param] = control

    substitute_values(template1,run1_filename,script1_changes)
    substitute_values(template2,run2_filename,script2_changes)


def run_simulation(simulation):
    # Run the simulation
    config = simulation["config"]
    i_iter = simulation["iteration"]
    run_command = config["run_parameters"]["run_command"]
    run_filename = f"{config['run_parameters']['run_filename_template']}{i_iter}.lammps"
    command = f"{run_command} --input {run_filename}"
    exit_code = os.system(command)


def run_differential_simulation(simulation):
    # Run the simulation
    config = simulation["config"]
    i_iter = simulation["iteration"]
    run1_command = config["run_parameters"]["run1_command"]
    run1_filename = f"{config['run_parameters']['run1_filename_template']}{i_iter}.lammps"
    command = f"{run1_command} --input {run1_filename}"
    exit_code = os.system(command)
    run2_command = config["run_parameters"]["run2_command"]
    run2_filename = f"{config['run_parameters']['run2_filename_template']}{i_iter}.lammps"
    command = f"{run2_command} --input {run2_filename}"
    exit_code = os.system(command)


# Create configuration with target cell size and density
def run_mc(simulation):
    config = simulation["config"]
    i_iter = simulation["iteration"]
    box = config["run_parameters"]["cell_size"]
    nmol = config["run_parameters"]["nmol"]
    npoly = config["run_parameters"]["npoly"]
    run_command = config["run_parameters"]["run_command"]
    in_filename = f"{config['run_parameters']['in_filename_template']}{i_iter}.data"
    run_filename = f"{config['run_parameters']['run_filename_template']}{i_iter}.lammps"
    mc_trials = config["run_parameters"]["mc_trials"]
    e_pairs = []
    for n_trial in range(mc_trials):
        #in_filename = NamedTemporaryFile(suffix = ".data")
        create_configuration(system_type = "random", npoly = npoly, nmol = nmol,
                             box = box, output = in_filename, add_angles = True,
                             add_dihedrals = False, self_avoid = 0.0,
                             atomtypes = None)
        command = f"{run_command} --input {run_filename}"
        exit_code = os.system(command)
        e_pair = e_pair_reader(file = "e_pair.dat")
        e_pairs.append(e_pair)
    mean_e_pair = np.mean(e_pairs)
    simulation["feedbacks"]["e_pair"] = mean_e_pair


def output_exists(simulation):
    config = simulation["config"]
    i_iter = simulation["iteration"]
    try:
        simulation_type = config["run_parameters"]["simulation_type"]
    except KeyError:
        simulation_type = "simple"
    if simulation_type == "simple":
        out_filename = f"{config['run_parameters']['out_filename_template']}{i_iter}.data"
        output_exists = os.path.isfile(out_filename)
    elif simulation_type == "differential":
        out1_filename = f"{config['run_parameters']['out1_filename_template']}{i_iter}.data"
        out2_filename = f"{config['run_parameters']['out2_filename_template']}{i_iter}.data"
        output1_exists = os.path.isfile(out1_filename)
        output2_exists = os.path.isfile(out2_filename)
        output_exists = output1_exists and output2_exists
    else:
        raise NameError("Unknown simulation type")
    return output_exists


def analyze_results(simulation):
    # Analyze result
    config = simulation["config"]
    i_iter = simulation["iteration"]
    has_output = output_exists(simulation)
    for param, details in config["model_parameters"].items():
        if has_output:
            feedback_name = details["output"]
            feedback = feedback_reader(feedback_name,
                                       details["output_args"])
            simulation["feedbacks"][feedback_name] = feedback
            print(f"Feedback {details['output']} \
                  is {simulation['feedbacks'][feedback_name]}")
        else:
            simulation["feedbacks"][details["output"]] = "n/a"
    simulation["output_exists"] = has_output


def write_log(simulation):
    config = simulation["config"]
    trials_file = config["run_parameters"]["log"]
    trial = {}
    trial["iteration"] = simulation["iteration"]
    trial.update(simulation["controls"])
    trial.update(simulation["feedbacks"])
    simulation["trials"].append(trial)
    trials_df = pd.DataFrame.from_records(simulation["trials"])
    if trials_file[-3:] == 'csv':
        trials_df.to_csv(trials_file)
    elif trials_file[-3:] == 'xls' or trials_file[-4:] == 'xlsx':
        trials_df.to_excel(trials_file)


def copy_data(simulation):
    config = simulation["config"]
    i_iter = simulation["iteration"]
    out_filename = f"{config['run_parameters']['out_filename_template']}{i_iter}.data"
    if simulation["output_exists"]:
        os.system(f"cp -a {out_filename} \
                  {config['run_parameters']['in_filename_template']}{i_iter+1}.data")
    else:
        raise NameError("Output doesn't exist")


def copy_differential_data(simulation):
    config = simulation["config"]
    i_iter = simulation["iteration"]
    out1_filename = f"{config['run_parameters']['out1_filename_template']}{i_iter}.data"
    out2_filename = f"{config['run_parameters']['out2_filename_template']}{i_iter}.data"
    if simulation["output_exists"]:
        os.system(f"cp -a {out1_filename} \
                  {config['run_parameters']['in1_filename_template']}{i_iter+1}.data")
        os.system(f"cp -a {out2_filename} \
                  {config['run_parameters']['in2_filename_template']}{i_iter+1}.data")
    else:
        raise NameError("Output doesn't exist")


def run_trial(simulation):
    config = simulation["config"]
    try:
        simulation_type = config["run_parameters"]["simulation_type"]
    except KeyError:
        simulation_type = "simple"

    if simulation_type == "simple":
        prepare_simulation(simulation)
        run_simulation(simulation)
        analyze_results(simulation)
        write_log(simulation)
        copy_data(simulation)

    elif simulation_type == "differential":
        prepare_differential_simulation(simulation)
        run_differential_simulation(simulation)
        analyze_results(simulation)
        write_log(simulation)
        copy_differential_data(simulation)
        
    elif simulation_type == "mc":
        prepare_simulation(simulation)
        run_mc(simulation)
        write_log(simulation)



# Read input parameters: filenames, PID variables, template
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Configuration Parameters")
    parser.add_argument("config", metavar = "JSON", type = str, nargs = '?',
    			default = "config.json", help = "study configuration file")
    args = parser.parse_args()

    config_filename = args.config

    with open(config_filename, "r") as f:
        config = json.load(f)
        
    n_iter = config["run_parameters"]["iterations"]
    
    simulation = {}
    
    simulation["config"] = config
    simulation["pids"] = {}
    simulation["trials"] = []
    simulation["feedbacks"] = {}
    
    timer = NCalls()
    
    simulation["timer"] = timer

    for i_iter in range(n_iter):
            simulation["iteration"] = i_iter
            run_trial(simulation)