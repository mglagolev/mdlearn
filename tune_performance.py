#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import os
import optuna
import pandas as pd
import argparse
import json


trials = []
trials_file = "trials.csv"


def parameter_ordered_names(parameters):
    p_names = list(parameters.keys())
    p_names.sort()
    return p_names


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


def performance_reader(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        if line[:12] == "Performance:":
            performance = float(line.split()[1])
            return performance


def builds_reader(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        if line[:20] == "Neighbor list builds":
            builds = int(line[20:].split()[1])
            return builds


def dangerous_builds_reader(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        if line[:16] == "Dangerous builds":
            builds = int(line[16:].split()[1])
            return builds


def warnings_reader(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    f.close()
    n_warnings = 0
    for line in lines:
        if line[:7] == "WARNING":
            n_warnings += 1
    return n_warnings


def objective(trial: optuna.Trial, config):
    try:
        args = []
        script_changes = []
        trial_data = { "ntrial" : trial.number }
        for param, details in config["model_parameters"].items():
            log = details.get('log', False)
            if log == "True":
                log = True
            if details['type'] == "int":
                value = trial.suggest_int(param, details['min'], details['max'], log=log)
            elif details['type'] == "float":
                value = trial.suggest_float(param, details['min'], details['max'], log=log)
            elif details['type'] == "categorical":
                if type(details['values']) == list:
                    value = trial.suggest_categorical(param, details['values'])
                elif type(details['values']) == dict:
                    value = trial.suggest_categorical(param, list(details['values'].keys()))
                else:
                    raise NameError("unsupported values data type")
            else:
                raise ValueError(f"Unsupported parameter type: {details['type']}")

            if details['type'] == "categorical" and type(details['values']) == dict:
                output_value = value
                value = details['values'][value]
            else:
                output_value = value

            if details['usage'] == 'script':
                script_changes.append((param, str(value)))
            elif details['usage'] == 'cli':
                args.append(f"--{param} {value}")
                
            trial_data[param] = output_value
            
            
        log_file_name = f"trial_{trial.number}.log"
        output_file_name = f"out_{trial.number}.data"

        script_changes.append(("LOG", log_file_name))
        script_changes.append(("OUTPUT", output_file_name))
        script_changes.append(("NTRIAL", str(trial.number)))


        executable = config["opt_parameters"]["executable"]
        trials_file = config["opt_parameters"]["log"]
        template = config["opt_parameters"]["template_filename"]
        run_filename_template = config["opt_parameters"]["run_filename_template"]
        run_filename = f"{run_filename_template}{trial.number}.lammps"
        substitute_values(template,run_filename,script_changes)
        
        command = f"{executable} --input {run_filename} {' '.join(args)}"

        exit_code = os.system(command)
        output_exists = os.path.isfile(output_file_name)
        if output_exists:
            good = True
            neighbor_count = builds_reader(log_file_name)
            dangerous_count = dangerous_builds_reader(log_file_name)
            warnings_count = warnings_reader(log_file_name)
            trial_data["neighbor_count"] = neighbor_count
            trial_data["dangerous_count"] = dangerous_count
            trial_data["warnings_count"] = warnings_count
            if neighbor_count >= 100:
                ratio = dangerous_count / neighbor_count
                if ratio > 0.01:
                    good = False
            elif neighbor_count > 0:
                if dangerous_count > 0:
                    good = False
            else:
                raise NameError("Neighbor list count appears to be 0")
            if warnings_count > 0:
                good = False
            if good:
                performance = performance_reader(log_file_name)
            else:
                performance = 0.0
        else:
            performance = 0.
            trial_data["neighbor_count"] = "n/a"
            trial_data["dangerous_count"] = "n/a"
            trial_data["warnings_count"] = "n/a"
            
        trial_data["performance"] = performance
        trials.append(trial_data)
        trials_df = pd.DataFrame.from_records(trials)
        
        if trials_file[-4:] == ".csv":
            trials_df.to_csv(trials_file)
        elif trials_file[-4:] == ".xls":
            trials_df.to_excel(trials_file)
        elif trials_file[-5:] == ".xlsx":
            trials_df.to_excel(trials_file)
        else:
            raise NameError("Unknown log file format")

        return performance

    except Exception as e:
        print(e)
        return 0.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Model Parameters")
    parser.add_argument("config", metavar = "JSON", type = str, nargs = '?',
    			default = "config.json", help = "study configuration file")
    args = parser.parse_args()
    
    config_filename = args.config
    
    with open(config_filename, "r") as f:
        config = json.load(f) 
    sampler_name = config["opt_parameters"]["sampler"]
    if sampler_name == "Random":
        sampler = optuna.samplers.RandomSampler()
    elif sampler_name == "TPE":
        sampler = optuna.samplers.TPESampler()
    else:
        raise NameError("unknown sampler")
    n_trials = config["opt_parameters"]["iterations"]
    study = optuna.create_study(direction="maximize", sampler = sampler)
    study.optimize(lambda trial: objective(trial, config), n_trials=n_trials, gc_after_trial=True)
    if True:
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        study.trials_dataframe().to_csv('study.csv')

