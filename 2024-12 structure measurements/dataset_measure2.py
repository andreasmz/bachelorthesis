# Libary to measure structures
# Adapted from IMB Summer School code
# Edited by Andreas Brilka 2024-12-16


from Bio.PDB import ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB import PDBParser
from chempy import cpv
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.bonds as bonds
import pathlib
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool
cpu_count = multiprocessing.cpu_count()


def calculate_buried_area(structure, chains, chain1, chain2)