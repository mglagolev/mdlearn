# mdlearn
Tools for introduction of machine learning to molecular dynamics simulations

Prerequisites:

	MDAnalysis (pip3 install MDAnalysis)
	mouse2 (pip3 install mouse2)

Usage:

Creating .pdb files for individual aggregates:

split_pdb.py [-h] [--r_neigh [R_neigh]] [--selection [QUERY]] INPUT [INPUT ...]

Determine clusters in the last time frame of a trajectory and write each one into a separate pdb file.

positional arguments:
  INPUT                input file(s), the format will be guessed by MDAnalysis based on file extension

options:
  -h, --help           show this help message and exit
  --r_neigh [R_neigh]  neighbor cutoff
  --selection [QUERY]  Consider only selected atoms, use MDAnalysis selection language

Example:

split_pdb.py somedata.data somedump.lammpsdump --selection "type 1"

PDB files named somedata-i.pdb will be generated, where i is the id of an aggregate, starting from 1.

Metadata manipulation:

rw_metadata.py [-h] [--phil [PHIL ...]] [--phob [PHOB ...]] FILE ACTION [type]

Add structure metadata to pdb

positional arguments:
  FILE               pdb file
  ACTION             write/read/clear
  type               Structure type

options:
  -h, --help         show this help message and exit
  --phil [PHIL ...]  Solvophilic bead types
  --phob [PHOB ...]  Solvophobic bead types


Example:

rw_metadata.py somedata-1.pdb write TORUS --phil 1 --phob 2

rw_metadata.py somedata-1.pdb read

rw_metadata.py somedata-1.pdb clear


