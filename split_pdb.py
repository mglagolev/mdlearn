#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:28:09 2023

@author: Mikhail Glagolev
"""
import os
import MDAnalysis as mda
from mouse2.lib.aggregation import determine_aggregates
from mouse2.lib.utilities import names_from_types

def main():
    
    import argparse

    parser = argparse.ArgumentParser(
        description =
        """Determine clusters in the last time frame of a trajectory and 
           write each one into a separate pdb file.""")

    parser.add_argument(
        'input', metavar = 'INPUT', action = "store", nargs = '+',
        help = """input file(s), the format will be guessed by MDAnalysis 
        based on file extension""")

    parser.add_argument(
        '--r_neigh', metavar = 'R_neigh', type = float, nargs = '?',
        default = 1.2, help = "neighbor cutoff")

    parser.add_argument(
        '--selection', metavar = 'QUERY', type = str, nargs = '?',
        help 
        = "Consider only selected atoms, use MDAnalysis selection language")
    
    parser.add_argument(
        "--remove-pbc-bonds", action = "store_true",
        help = "Remove the bonds transversing the periodic boundary conditions")
    
    parser.add_argument(
        "--bonded-as-neighbors", action = "store_true",
        help = "Treat bonded atoms as each other neighbors regardless of"
                + " the distance between them")

    args = parser.parse_args()

# Read data
    u = mda.Universe(*args.input)

# Remove the bonds transversing the PBC, if necessary

    if args.remove_pbc_bonds:
        minbox = min(u.dimensions) / 2.
        bonds_to_delete = [
            bond for bond in u.bonds if bond.length(pbc = False) > minbox]
        u.delete_bonds(bonds_to_delete)

# Write atom name attribute for pdb
    names_from_types(u)

# Determine aggregates
    data = determine_aggregates(u, r_neigh = args.r_neigh,
                                selection = args.selection, ts_indices = [-1],
                                bonded_as_neighbors = args.bonded_as_neighbors)

    aggregates = list(data["data"].values())[0]

    for i in range(len(aggregates)):
        aggregate_atoms = u.select_atoms("index "
                                         + " ".join(map(str, aggregates[i])))
        output = os.path.splitext(args.input[0])[0] + "-" + str(i+1) + ".pdb"
        aggregate_atoms.write(output)        
    
if __name__ == "__main__":
    main()