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

def c_ratio_reader(data = None, atomdump = None):
    import MDAnalysis as mda
    from microplastic.c_ratio import c_ratio
    u = mda.Universe(data, atomdump)
    c_ratio, err = c_ratio(u)
    return c_ratio

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

    template = config["run_parameters"]["template_filename"]

    run_filename_template = config["run_parameters"]["run_filename_template"]

    in_filename_template = config["run_parameters"]["in_filename_template"]

    out_filename_template = config["run_parameters"]["out_filename_template"]

    run_command = config["run_parameters"]["run_command"]
    
    trials_file = config["run_parameters"]["log"]

    pids = {}
    
    feedbacks = {}
    
    trials = []
    
    timer = NCalls()

    for i_iter in range(n_iter):
# Prepare the simulation
        in_filename = f"{in_filename_template}{i_iter}.data"
        out_filename = f"{out_filename_template}{i_iter}.data"
        run_filename = f"{run_filename_template}{i_iter}.lammps"
        script_changes = []
        script_changes.append(["INPUT", in_filename])
        script_changes.append(["OUTPUT", out_filename])
        print(f"Step {i_iter}")
        trial = {}
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
                #control = math.exp(control)
                print(f"Control {param} is {control}")
            trial[param] = control
            script_changes.append([param, str(control)])

        substitute_values(template,run_filename,script_changes)
# Run the simulation
        command = f"{run_command} --input {run_filename}"
        exit_code = os.system(command)
        output_exists = os.path.isfile(out_filename)
# Analyze result
        for param, details in config["model_parameters"].items():
            if output_exists:
                feedback_name = details["output"]
                feedback = feedback_reader(feedback_name,
                                                      details["output_args"])
                feedbacks[feedback_name] = feedback
                trial[details["output"]] = feedback
                print(f"Feedback {details['output']} is {feedbacks[feedback_name]}")
            else:
                trial[details["output"]] = "n/a"
        trials.append(trial)
        trials_df = pd.DataFrame.from_records(trials)
        if trials_file[-3:] == 'csv':
            trials_df.to_csv(trials_file)
        elif trials_file[-3:] == 'xls' or trials_file[-4:] == 'xlsx':
            trials_df.to_excel(trials_file)
        if output_exists:
            os.system(f"cp -a {out_filename} {in_filename_template}{i_iter+1}.data")
        else:
            raise NameError("Output doesn't exist")