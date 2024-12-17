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
from multiprocessing import Process, Queue
import queue
cpu_count = multiprocessing.cpu_count()
import time
from multiprocessing import Pool



# Function to calculate buried area using Shrake-Rupley
def calculate_buried_area(structure):
    chains = [c for c in structure.get_chains()]
    assert len(chains) == 2

    # Calculate SASA for the whole structure
    sasa = ShrakeRupley()
    sasa.compute(structure, level="A")
    total_area = np.sum(atom.sasa for atom in structure.get_atoms())

    # Calculate buried area for each chain separately
    chain1 = structure[0][chains[0].id]
    chain2 = structure[0][chains[1].id]
    
    sasa.compute(chain1, level="A")
    area_ch1 = np.sum(atom.sasa for atom in chain1.get_atoms())
    
    sasa.compute(chain2, level="A")
    area_ch2 = np.sum(atom.sasa for atom in chain2.get_atoms())

    # Calculate buried area
    buried_area = (area_ch1 + area_ch2 - total_area)
    return round(buried_area, 3)
 
# Function to calculate the minimum distance of interface residues
def minimum_interface_distance(structure):
    chains = [c for c in structure.get_chains()]
    assert len(chains) == 2

    chain1 = structure[0][chains[0].id]
    chain2 = structure[0][chains[1].id]

    # In contrast to original code, I needed to remove the constraints to interface atom only
    # But it shouldn't matter, right?

    min_distance = float('inf')  # Initialize with a large number

    atomsCA_chain1 = [a for a in chain1.get_atoms() if a.get_name() == "CA"]
    atomsCA_chain2 = [a for a in chain2.get_atoms() if a.get_name() == "CA"]

    for atom1 in atomsCA_chain1:
        for atom2 in atomsCA_chain2:
            min_distance = min(min_distance, cpv.distance(atom1.coord, atom2.coord))

    # If no distances were found, return 0, otherwise return the minimum distance
    return min_distance if min_distance != float('inf') else 0


# Function to calculate hydrogen bonds
def find_h_bonds(structure, structure_biotite):
    chains = [c for c in structure.get_chains()]
    assert len(chains) == 2
    
    # If the structure contains only a single model, convert AtomArrayStack to AtomArray
    if structure_biotite.stack_depth() == 1:
        atom_array = structure_biotite[0]
    else:
        raise ValueError("The provided structure contains multiple models. Please provide a structure with a single model.")
    
    # Generate a BondList using a distance-based approach
    bond_list = bonds.connect_via_distances(atom_array)

    # Assign the generated BondList to the atom array
    atom_array.bonds = bond_list

    # Create selection masks for the two chains
    selection1 = atom_array.chain_id == chains[0].id
    selection2 = atom_array.chain_id == chains[1].id
    
    # Calculate hydrogen bonds between the two chains
    try:
        triplets = struc.hbond(atom_array, selection1=selection1, selection2=selection2)
        if isinstance(triplets, tuple):
            triplets = triplets[0]  # Extract the first item if it's a tuple
    except Exception as e:
        print(f"Error calculating hydrogen bonds: {e}")
        return 0

    return len(triplets)

# Function to calculate salt bridges 
def find_salt_bridges(structure, cutoff=4.0):
    chains = [c for c in structure.get_chains()]
    assert len(chains) == 2

    chain1 = structure[0][chains[0].id]
    chain2 = structure[0][chains[1].id]

    saltBridges_ac = {"ASP":"a", "GLU":"a", "ARG":"b", "LYS":"b"}
    saltBridges_atoms = ['OD1', 'OD2', 'OE1', 'OE2', 'NH1', 'NH2', 'NE', 'NZ']

    salt_bridges = 0

    # In the original code, salt bridges were only found with acidics in chain1. Not it is also with basics in chain1
    for res1 in chain1:
        if res1.resname not in saltBridges_ac.keys():
            continue
        for res2 in chain2:
            if res2.resname not in saltBridges_ac.keys():
                continue
            if saltBridges_ac[res1.resname] == saltBridges_ac[res2.resname]:
                continue
            for atom1 in [a for a in res1 if a.id in saltBridges_atoms]:
                for atom2 in [a for a in res2 if a.id in saltBridges_atoms]:
                    distance = atom1 - atom2
                    if distance <= cutoff:
                        salt_bridges += 1

    return salt_bridges

# Function to calculate hydrophobic interactions considering only carbon atoms
def find_hydrophobic_interactions(structure, cutoff=5.0):
    chains = [c for c in structure.get_chains()]
    assert len(chains) == 2

    # Get the chains
    chain1 = structure[0][chains[0].id]
    chain2 = structure[0][chains[1].id]

    hydrophobic_interactions = 0

    # Hydrophobic residues including Glycine (GLY) and Proline (PRO)
    hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'PRO', 'TRP', 'GLY'}

    # Compare each hydrophobic residue in chain1 with each hydrophobic residue in chain2
    for res1 in [r for r in chain1 if r.resname in hydrophobic_residues]:
        for atom1 in [a for a in res1 if a.element == 'C']:
            for res2 in [r for r in chain2 if r.resname in hydrophobic_residues]:
                for atom2 in [a for a in res2 if a.element == 'C']:
                    distance = atom1 - atom2
                    if distance <= cutoff:
                        hydrophobic_interactions += 1

    
    return hydrophobic_interactions

parser = PDBParser(QUIET=True)

def evaluate_structure(path: pathlib.Path, structure_name: str) -> dict|None:
    filename = path.stem
    try:
        structure = parser.get_structure("structure", file=path)
        structure_biotite = pdb.get_structure(pdb.PDBFile.read(path))
    except PDBConstructionException:
        print(f"Can't parse structure {structure_name} (file {path.stem})")
        return None
    except ValueError as ex:
        print(f"Can't parse structure {structure_name} (file {filename}) due to the following reason: {ex}")
        return None
    chains = [c for c in structure.get_chains()]
    if len(chains) != 2:
        print(f"Can't parse structure {structure_name} (file {filename}) because it has not 2 chains")
        return None

    buried_area = calculate_buried_area(structure)
    hbonds = find_h_bonds(structure[0], structure_biotite)
    min_distance = minimum_interface_distance(structure)
    salt_bridges = find_salt_bridges(structure)
    hydrophobic_interactions = find_hydrophobic_interactions(structure)
    assert type(structure_name) == str
    assert type(filename) == str
    assert type(hbonds) == int
    assert type(salt_bridges) == int
    assert type(min_distance) == float
    assert type(hydrophobic_interactions) == int
    return {
        'prediction_name': structure_name,
        'structure_file': filename,
        'hbonds': hbonds,
        'salt_bridges': salt_bridges,
        'buried_area': buried_area,
        'min_distance': min_distance,
        'hydrophobic_interactions': hydrophobic_interactions
    }
    

def readFolder(structure_basePath, structure_folders):
    structures_count = 0
    pdb_structures_path = {}

    for folder in structure_folders:
        folder_path = pathlib.Path.absolute(structure_basePath / folder)
        if not folder_path.is_dir():
            print(f"\tERROR: {folder_path} is not a folder")
            continue
        pdb_structures_path[folder] = {}
        for prediction_path in folder_path.iterdir():
            if not prediction_path.is_dir():
                continue
            structure_name = prediction_path.stem
            pdb_structures_path[folder][structure_name] = {}
            for pdb_file in prediction_path.iterdir():
                if not str(pdb_file).endswith(".pdb"):
                    continue
                structures_count+=1
                pdb_structures_path[folder][structure_name][pdb_file.stem] = pdb_file
    print(f"Found {structures_count} structures")
    return pdb_structures_path

def evaluteFolder(pdb_structures_path, silent=True, num_threads=cpu_count):
    #tasks_queued = Queue()
    #tasks_finished = Queue()
    #tasks_finished.cancel_join_thread()
    tasks = []
    for folder, structureDict in pdb_structures_path.items():
        for structure_name, fileArray in structureDict.items():
            for file_name, path in fileArray.items():
                tasks.append([path, structure_name])
                #tasks_queued.put([path, structure_name])
    results = runQueue(tasks, silent=silent, num_threads=num_threads)
    print(len(results))

    #runQueue(tasks_queued, tasks_finished, silent=silent, num_threads=num_threads)
    #results = []
    #print(tasks_finished.qsize())
    #while not tasks_finished.empty():
    #    task = tasks_finished.get()
    #    results.append(task)
    #print(len(results))
    results = [x for x in results if x is not None]
    if len(results) == 0:
        return None
    return pd.DataFrame(results).sort_values(["prediction_name", "structure_file"])

"""
def _run_task(tasks_queued: Queue, tasks_finished: Queue, silent=True):
    tasks_finished.cancel_join_thread()
    while tasks_queued.qsize() != 0:
        if (tasks_queued.qsize() % 25 == 0):
            print(f"{tasks_queued.qsize()} queued")
        try:
            task = tasks_queued.get(timeout=3)
        except queue.Empty:
            print("Empty error")
            continue
        else:
            r = evaluate_structure(*task)
            if r is not None and not silent:
                print(r["prediction_name"], r["structure_file"])
            tasks_finished.put(r)
            #if r is None:
            #    tasks_finished.put(None)
            #else:
            #    tasks_finished.put(("abcs"))
    return True
"""

def _run_task(tasks):
    silent, task = tasks[0], tasks[1:]
    r = evaluate_structure(*task)
    if r is not None and not silent:
        print(r["prediction_name"], r["structure_file"])
    return r


#def runQueue(tasks_queued: Queue, tasks_finished: Queue, silent=True, num_threads=cpu_count):
def runQueue(tasks: list, silent=True, num_threads=cpu_count):
    """
    processes = []
    for w in range(num_threads):
        p = Process(target=_run_task, args=(tasks_queued, tasks_finished, silent))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Process {p.name} exited with {p.exitcode}")
    """
    tasks = [[silent, *t] for t in tasks]

    p = Pool(processes=num_threads)
    data = p.map(_run_task, tasks)
    p.close()
    return data

if __name__ == "__main__":

    structure_basePath = pathlib.Path("ressources/ISS AF_DMI_structures").absolute()
    outputPath = pathlib.Path(r"2024-12-16/output/structures_measured.csv").absolute()
    structure_folders = ['AF_DMI_structures1', 'AF_DMI_structures2', 'AF_DMI_structures3']

    pdb_structures_path = readFolder(structure_basePath, structure_folders)
    results = evaluteFolder(pdb_structures_path, silent=False)
    
    if results is not None:
        results.to_csv(outputPath, index=False)
