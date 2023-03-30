#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:27:13 2023

@author: Mikhail Glagolev
"""

def pdb_write_structure(filename, structure = None, solvophobic = [], 
                        solvophilic = []):
    """
    Write structure metadata to pdb file

    """
    delimiter = " "
    pdb_file = open(filename, mode = 'r')
    pdb_lines = pdb_file.readlines()
    pdb_file.close()
    key = "REMARK 250 "
    lines = []
    lines.append(key + "EXPERIMENTAL DETAILS\n")
    lines.append(key + "STRUCTURE TYPE: " + str(structure) + "\n")
    lines.append(key + "SOLVOPHOBIC UNITS: " + 
                 delimiter.join(map(str, solvophobic)) + "\n")
    lines.append(key + "SOLVOPHILIC UNITS: " +
                 delimiter.join(map(str, solvophilic)) + "\n")
    if max(map(len, lines)) > 80:
        raise(NameError("Line too long for pdb file"))
    pdb_file = open(filename, mode = 'w')
    pdb_file.writelines(lines)
    pdb_file.writelines(pdb_lines)
    pdb_file.close()
    

def pdb_read_structure(filename):
    """
    Read structure metadata from pdb file

    """
    result = {}
    pdb_file = open(filename, mode = 'r')
    for line in pdb_file.readlines():
        line = line.strip()
        if line[:10] == "REMARK 250":
            if line[11:26] == "STRUCTURE TYPE:":
                result["structure"] = line[27:]
            elif line[11:29] == "SOLVOPHOBIC UNITS:":
                result["solvophobic"] = line[30:].strip().split()
            elif line[11:29] == "SOLVOPHILIC UNITS:":
                result["solvophilic"] = line[30:].strip().split()
    return result

def pdb_clear_structure(filename):
    """
    Remove structure metadata from pdb file
    """
    pdb_file = open(filename, mode = 'r')
    lines = pdb_file.readlines()
    pdb_file.close()
    pdb_file = open(filename, mode = 'w')
    for line in lines:
        if line[:10] != "REMARK 250":
            pdb_file.write(line)
    pdb_file.close()

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description = 'Add structure metadata to pdb')

    parser.add_argument(
        'filename', metavar = 'FILE', action = "store",
        help = "pdb file")
    
    parser.add_argument(
        'action', metavar = 'ACTION', action = "store",
        help = "write/read/clear")
    
    parser.add_argument(
        'type', nargs = '?', default = None, 
        type = str, help = "Structure type")
    
    parser.add_argument(
        '--phil', nargs = '*', 
        help = "Solvophilic bead types")
    
    parser.add_argument(
        '--phob', nargs = '*', 
        help = "Solvophobic bead types")
    
    args = parser.parse_args()
    
    if args.action == "write":
        if args.type == None:
            raise NameError("A structure type must be specified")
        if len(args.phil) == 0 or len(args.phob) == 0:
            raise NameError(
                "Solvophilic and solvophobic unit types must be specified")
        pdb_write_structure(args.filename, structure = args.type,
                            solvophobic = args.phil,
                            solvophilic = args.phob)
    
    elif args.action == "read":
        data = pdb_read_structure(args.filename)
        sys.stderr.write(str(data) + "\n")
    
    elif args.action == "clear":
        pdb_clear_structure(args.filename)

if __name__ == "__main__":
    main()