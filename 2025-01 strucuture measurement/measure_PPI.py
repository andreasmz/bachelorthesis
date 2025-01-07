# created by Andreas Brilka from a code basis from IMB summmer school
# 2024-12-16

import math
import numpy as np
import pathlib
import pandas as pd
import sys
import time
from multiprocessing import Pool, cpu_count
import logging

cpu_count = cpu_count()
logger = logging.getLogger("measure_PPI") # For general logging
if len(logger.handlers) == 0:
    formatter = logging.Formatter(fmt="[%(asctime)s | %(module)s | %(levelname)s] %(message)s")
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(streamHandler)
if __name__ == "__main__":
    logger.info("Loaded measure_PPI libary")

import biotite.structure as struc
import biotite.structure.io.pdb as bt_pdb
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure as BioPy_PDBStructure
from Bio.PDB.PDBExceptions import PDBConstructionException

parser = PDBParser(QUIET=True)

def OpenStructure(path: pathlib.Path, structure_name: str = "") -> tuple[BioPy_PDBStructure|None, struc.AtomArray|None]:
    """
        Opens the given structure and returns the Bio.PDB and biotite objects.
    """
    t0 = time.perf_counter()
    file_name = path.name
    try:
        structure_biopy = parser.get_structure("structure", file=path)
        structure_biotite = bt_pdb.get_structure(bt_pdb.PDBFile.read(path))
    except PDBConstructionException:
        logger.warning(f"Can't parse structure {structure_name} (file {file_name}) using Biopython")
        return (None, None)
    except ValueError as ex:
        logger.warning(f"Can't parse structure {structure_name} (file {file_name}) due to the following reason: {ex}")
        return (None, None)
    
    if structure_biotite.stack_depth() != 1:
        logger.warning(f"Can't parse structure {structure_name} (file {file_name}) because it contains more than one stack")
        return (None, None)
    
    atomarray_biotite: struc.AtomArray = structure_biotite[0]

    chains = [c for c in structure_biopy.get_chains()]
    if len(chains) != 2:
        logger.warning(f"Can't parse structure {structure_name} (file {file_name}) because it has not 2 chains")
        return (None, None)
    
    t1 = time.perf_counter()
    logger.debug(f"Runtime reading structure {structure_name} (file {file_name}): {round((t1-t0)*1000, 1)}ms")
    return (structure_biopy, atomarray_biotite)


def calculate_buried_area(atomarray_biotite:struc.AtomArray, probe_radius:float=1.4):
    """
        Calculates the buried surface area using biotite which is defined as surface area of the two chains
        subtracted from the surface area of the complex.
    """
    ti = time.perf_counter()
    chains = struc.get_chains(atomarray_biotite)
    assert len(chains) == 2

    chain1 = atomarray_biotite[atomarray_biotite.chain_id == chains[0]]
    chain2 = atomarray_biotite[atomarray_biotite.chain_id == chains[1]]
    t1 = time.perf_counter()

    sasa12 = np.sum([s for s in struc.sasa(atomarray_biotite, probe_radius=probe_radius) if math.isfinite(s)])
    sasa1 = np.sum([s for s in struc.sasa(chain1, probe_radius=probe_radius) if math.isfinite(s)])
    sasa2 = np.sum([s for s in struc.sasa(chain2, probe_radius=probe_radius) if math.isfinite(s)])
    logger.debug(f"Sasa values: Chain 1 = {round(sasa1, 3)}, Chain 2 = {round(sasa2, 3)}, Total = {round(sasa12, 3)}")
    buried_area = (sasa1 + sasa2 - sasa12)
    tf = time.perf_counter()
    logger.debug(f"Runtime calculate_buried_area: {round((tf-ti)*1000, 1)}ms ({round((t1-ti)*1000, 1)}ms generating chains, {round((tf-t1)*1000, 1)}ms sasa)")
    return round(buried_area, 3)


def calculate_min_distance(atomarray_biotite:struc.AtomArray):
    """
        Calculates the minimum distance between the two chains of a protein complex using biotite.
        The minimum distance is defined as the distance between the backbone (CA atoms)
    """
    ti = time.perf_counter()
    chains = struc.get_chains(atomarray_biotite)
    assert len(chains) == 2

    chain1 = atomarray_biotite[atomarray_biotite.chain_id == chains[0]]
    chain2 = atomarray_biotite[atomarray_biotite.chain_id == chains[1]]
    t1 = time.perf_counter()

    chain1_backbone = chain1[chain1.atom_name == "CA"]
    chain2_backbone = chain2[chain2.atom_name == "CA"]
    min_distance = float("inf")
    for a1 in chain1_backbone:
        for a2 in chain2_backbone:
            min_distance = min(min_distance, struc.distance(a1, a2))
    tf = time.perf_counter()
    logger.debug(f"Runtime calculate_min_distance: {round((tf-ti)*1000, 1)}ms ({round((t1-ti)*1000, 1)}ms generating chains, {round((tf-t1)*1000, 1)}ms calculating distance)")
    
    return round(min_distance, 3) if math.isfinite(min_distance) else float('NaN')

def calculate_hbonds(atomarray_biotite:struc.AtomArray):
    """
        Calculates the number of hbonds between two chains of a protein complex using biotites AtomArray
    """
    ti = time.perf_counter()
    chains = struc.get_chains(atomarray_biotite)
    assert len(chains) == 2

    chain1_mask = atomarray_biotite.chain_id == chains[0]
    chain2_mask = atomarray_biotite.chain_id == chains[1]
    t1 = time.perf_counter()

    bond_list = struc.bonds.connect_via_distances(atomarray_biotite)
    atomarray_biotite.bonds = bond_list

    t2 = time.perf_counter()

    triplets = struc.hbond(atomarray_biotite, selection1=chain1_mask, selection2=chain2_mask)
    tf = time.perf_counter()
    logger.debug(f"Runtime calculate_hbonds: {round((tf-ti)*1000, 1)}ms ({round((t1-ti)*1000, 1)}ms generating chains, {round((t2-t1)*1000, 1)}ms bond list, {round((tf-t2)*1000, 1)}ms hbonds)")
    
    return triplets.shape[0]


def calculate_saltbridges(structure_biopy:BioPy_PDBStructure, cutoff:float=4.0):
    """
        Calculates the number of saltbridges between the two chains of a protein complex using biopython.
        Saltbridges are defined as a interaction between an acidic residue (ASP, GLU) with a basic residue 
        (ARG, LYS) and found if the distance between the oxygen and nitrogen atoms is below cutoff [Angstrom]
    """
    ti = time.perf_counter()
    chains = [c for c in structure_biopy.get_chains()]
    assert len(chains) == 2

    chain1 = structure_biopy[0][chains[0].id]
    chain2 = structure_biopy[0][chains[1].id]

    saltBridges_ac = {"ASP":"a", "GLU":"a", "ARG":"b", "LYS":"b"} # a: Acidic, b: Basic
    saltBridges_atoms = ['OD1', 'OD2', 'OE1', 'OE2', 'NH1', 'NH2', 'NE', 'NZ'] # 0,1: ASP, 2,3: GLU, 4,5,6: ARG, 7: LYS

    salt_bridges = 0

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
    tf = time.perf_counter()
    logger.debug(f"Runtime calculate_saltbridges: {round((tf-ti)*1000, 1)}ms")
    
    return salt_bridges

def calculate_hydrophobic_interactions(structure_biopy:BioPy_PDBStructure, cutoff:float=5.0):
    """
        Calculates the number of hydrophobic interactions between two chains of a protein complex using biopython.
        Hydrophobic interactions are defined if the C atoms are below the cutoff value [Angstrom] of the following
        residues: ALA, VAL, LEU, ILE, MET, PHE, PRO, TRP, GLY
    """
    ti = time.perf_counter()
    chains = [c for c in structure_biopy.get_chains()]
    assert len(chains) == 2

    chain1 = structure_biopy[0][chains[0].id]
    chain2 = structure_biopy[0][chains[1].id]

    hydrophobic_interactions = 0

    hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'PRO', 'TRP', 'GLY'}

    # Compare each hydrophobic residue in chain1 with each hydrophobic residue in chain2
    for res1 in [r for r in chain1 if r.resname in hydrophobic_residues]:
        for atom1 in [a for a in res1 if a.element == 'C']:
            for res2 in [r for r in chain2 if r.resname in hydrophobic_residues]:
                for atom2 in [a for a in res2 if a.element == 'C']:
                    distance = atom1 - atom2
                    if distance <= cutoff:
                        hydrophobic_interactions += 1

    tf = time.perf_counter()
    logger.debug(f"Runtime calculate_hydrophobic_interactions: {round((tf-ti)*1000, 1)}ms")
    
    return hydrophobic_interactions




def EvaluateStructure(path: pathlib.Path, structure_name: str = "") -> dict|None:
    """
        Measures the pdb file given by path
    """
    ti = time.perf_counter()
    file_name = path.name
    structure_biopy, atomarray_biotite = OpenStructure(path, structure_name)
    if structure_biopy is None or atomarray_biotite is None: return None

    buried_area = calculate_buried_area(atomarray_biotite)
    hbonds = calculate_hbonds(atomarray_biotite)
    min_distance = calculate_min_distance(atomarray_biotite)
    salt_bridges = calculate_saltbridges(structure_biopy)
    hydrophobic_interactions = calculate_hydrophobic_interactions(structure_biopy)

    tf = time.perf_counter()
    logger.info(f"parsed {structure_name} (file {file_name}) in {round((tf-ti), 3)}s")
    return {
        'structure_name': structure_name,
        'file': file_name,
        'hbonds': hbonds,
        'salt_bridges': salt_bridges,
        'buried_area': buried_area,
        'min_distance': min_distance,
        'hydrophobic_interactions': hydrophobic_interactions
    }


def _run_task(task):
        r = EvaluateStructure(*task)
        return r

def Run(pathObj: list[tuple[pathlib.Path, str]], num_threads=cpu_count) -> pd.DataFrame|None:
    """
        Measures the given paths and returns the result as pandas Dataframe.
        pathObj is a list of tuples of (path_to_pdf: pathlib.Path, structure_name: str)
        [The structurename is used for the output as filenames often are not unique]
    """
    logger.info(f"Started Taskpool of {num_threads} processes for {len(pathObj)} files")
    p = Pool(processes=num_threads)
    results = []
    _t = time.perf_counter()
    for i, r in enumerate(p.imap(_run_task, pathObj)):
        if i % 25 == 0:
            _speed = ((time.perf_counter() - _t)/25)**-1
            logger.info(f"parsed {int(100*i/len(pathObj))}% with a speed of {round(_speed, 3)} s⁻¹")
            _t = time.perf_counter()
        if r is not None:
            results.append(r)
    #results = p.map(_run_task, pathObj)
    p.close()

    results = [x for x in results if x is not None]
    if len(results) == 0:
        return None
    return pd.DataFrame(results).sort_values(["structure_name", "file"])

def WalkFolder(basePath: str, 
               pathObj:dict[str, dict[str, pathlib.Path]]={},
               structures: None|str|list[str] = None,
               files: None|bool|str|list[str] = None
               ) -> dict[str, dict[str, pathlib.Path]]:
    """
        Add the path basePath/structure/file.pdb to the pathObj provided (or create a new one if omitted).
        If files and/or structures are None, search inside the directory for all pdb files.
        Returns:
            pathObj: dict[name:str, tuple[path: pathlib.Path, structure_name: str]]
    """

    structures_count = 0
    basePath = pathlib.Path(basePath).absolute()
    if not basePath.is_dir():
        raise ValueError("The given basePath is not a valid directory")
    
    if structures is None:
        structures: list[pathlib.Path] = [p for p in basePath.iterdir()]
    elif isinstance(structures, str):
        structures: list[pathlib.Path] = [basePath / structures]
    elif isinstance(structures, list):
        structures: list[pathlib.Path] = [basePath / p for p in structures]
    else:
        raise ValueError("Invalid argument for structures")

    for structure in structures:
        if not structure.exists():
            raise ValueError(f"The structure {structure} does not point to a valid path")
        structure_name = str(structure.name)
        if structure.is_file():
            if structure.suffix.lower() == ".pdb":
                structure_name = str(structure.stem)
                if structure_name in pathObj.keys():
                    raise ValueError(f"Duplicate structure and file {structure}")
                pathObj[structure_name] = (structure.absolute(), structure_name)
                structures_count += 1
            continue

        if files is None:
            filesF: list[pathlib.Path] = [f for f in structure.iterdir() if f.is_file()]
        elif isinstance(files, str):
            filesF: list[pathlib.Path] = [structure / f"{files}.pdb"]
        elif isinstance(files, list):
            filesF: list[pathlib.Path] = [structure / f"{f}.pdb" for f in files]
        else:
            raise ValueError("Invalid argument for files")
        
        for file in filesF:
            if not file.exists() or not file.is_file():
                raise ValueError(f"{structure}/{file} does not point to a valid file")
            if not file.suffix.lower() == ".pdb":
                continue
            file_name = file.stem
            name = f"{structure_name}-{file_name}"
            if name in pathObj.keys():
                raise ValueError(f"Duplicate structure and file {structure}/{file_name}.pdb")
            pathObj[name] = (file.absolute(), structure_name)
            structures_count += 1
    print(f"Found {structures_count} structures")
    return pathObj

if __name__ == "__main__":
    structure_basePath = (pathlib.Path(__file__) / ".." / ".." / "ressources" / "ISS AF_DMI_structures").resolve()
    structure_folders = [structure_basePath / p for p in ['AF_DMI_structures1', 'AF_DMI_structures2', 'AF_DMI_structures3']]
    pathObj = WalkFolder(structure_folders[1])
    Run(list(pathObj.values()))