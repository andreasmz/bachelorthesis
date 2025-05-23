# created by Andreas Brilka from a code basis from IMB summmer school
# 2024-12-16

__version__ = "1.0.3"

import datetime
import math
import numpy as np
import pathlib
import pandas as pd
import sys
import time
import logging
from io import StringIO
from multiprocessing import Pool, cpu_count, get_logger
import multiprocessing_logging
multiprocessing_logging.install_mp_handler()
from typing import Self

import biotite.structure as struc
import biotite.structure.io.pdb as bt_pdb
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure as BioPy_PDBStructure
from Bio.PDB.Model import Model as BioPy_PDBModel
from Bio.PDB.PDBExceptions import PDBConstructionException

LOGLEVEL_ADDITIONAL_INFO = 19 # The module logs more information like for example the current processed file with this level

logger = logging.getLogger("measure_PPI")
formatter = logging.Formatter(fmt="[%(asctime)s | %(module)s | %(levelname)s] %(message)s")
streamHandler = logging.StreamHandler(sys.stdout)
streamHandler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(streamHandler)
logging.addLevelName(LOGLEVEL_ADDITIONAL_INFO, "INFO")

class Stopwatch:
    """
        A helper class to measure performance. Supports laps and nested stopwatches and is able to return average runtimes for all the stopwatches supplied.
    """

    def __init__(self):
        self._times = {}
        self.times = None
        self._stopwatches = {}
    def Start(self) -> Self:
        """
            Starts the stopwatch
        """
        self._times["initial"] = time.perf_counter()
        return self

    def Lap(self, name:str, stopwatch:Self=None) -> Self:
        """
            Create a lap for the current stopwatch = time since last lap call.
            If a stopwatch is provided, it will be parsed as a subentry when calling Stopwatch.Evaluate()
        """
        if "initial" not in self._times.keys(): raise RuntimeError("Please first initialize the Stopwatch by Starting it")
        if name == "final" or name=="initial": raise ValueError(f"'{name}' is an invalid name")
        if name in self._times.keys(): raise ValueError("Duplicate key entry for lap name")

        self._times[name] = time.perf_counter()

        if stopwatch is not None:
            self._stopwatches[name] = stopwatch
        return self

    def Stop(self) -> Self:
        """
            Stops the stopwatch
        """
        if "initial" not in self._times.keys():
            raise RuntimeError("Please first initialize the Stopwatch by Starting it")
        self._times["final"] = time.perf_counter()
        self.times = {}
        for i in range(1, len(self._times)):
            ks = list(self._times.keys())
            t_n1 = self._times[ks[i-1]]
            t_n = self._times[ks[i]]
            name_n = ks[i]
            if name_n == "final": continue
            self.times[name_n] = t_n - t_n1
        self.times["total runtime"] = self._times["final"] - self._times["initial"]
        return self

    def Evaluate(stopwatches: list[Self]):
        """
            Evaluate the list of stopwatches provided and return a dict of average runtimes.

            Example output (all times in rounded milliseconds)

            {
                'Step 1': (n=13, mean=23.55, var=2.34),
                'Step 2': {
                    'Substep 1': (n=7, mean=0.33, var=0.0),
                    'Substep 2': (n=7, mean=10.9, var=9.3),
                    'Total': (n=7, mean=11.3, var=7.4)
                }
                'Total': (n=13, mean=34.34, var=5.22)
            }
        """
        runtimes: dict[str,list[float]] = {}
        for s in stopwatches:
            for k, v in s.times.items():
                runtimes.get(k, [])
                if k in s._stopwatches.keys():
                    runtimes[k].append(v)
                    runtimes[k] = Stopwatch.Evaluate(s._stopwatches)
                else:
                    runtimes[k].append(v)
        runtime_stats = {k:(len(v), np.mean(v), np.var(v)) for k,v in runtimes.items() if type(v) == tuple}
        return runtime_stats

_freesasa_ready = False
try:
    import freesasa
    _freesasa_ready = True
except ModuleNotFoundError:
    logger.warning("You don't have freesasa installed. Falling back to biotite")

parser = PDBParser(QUIET=True)

# This function is included to allow worker threads to output to Jupyter Notebooks as multiprocessing.Pool does not allow to alter sys.stdout of the child processes.
# Therefore this function redirects the log output to a buffer, which is transmitted back to the main thread where it is outputed.
def _WorkerThreadIni(logLevel:int):
    """
        This function is called in the worker threads by multiprocessing.Pool
    """
    global stdout, logger
    stdout = StringIO()
    sys.stdout = stdout
    streamHandler.setStream(sys.stdout)
    logger.setLevel(logLevel)

class ProteinStructureWarning(Exception):
    def __init__(self, message):            
        super().__init__(message)

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

def calculate_buried_area(structure_biopy:BioPy_PDBStructure):
    """
        Calculates the buried surface area using freesasa which is defined as surface area of the two chains
        subtracted from the surface area of the complex.
    """
    ti = time.perf_counter()
    chains = [c for c in structure_biopy.get_chains()]
    assert len(chains) == 2

    chain1 = structure_biopy[0][chains[0].id]
    chain2 = structure_biopy[0][chains[1].id]

    strucChain1 = BioPy_PDBStructure('structure')
    modelChain1 = BioPy_PDBModel("1")
    modelChain1.add(chain1)
    strucChain1.add(modelChain1)
    strucChain2 = BioPy_PDBStructure('structure')
    modelChain2 = BioPy_PDBModel("1")
    modelChain2.add(chain2)
    strucChain2.add(modelChain2)
    t1 = time.perf_counter()

    fs_pp = freesasa.structureFromBioPDB(structure_biopy)
    fs_chain1 = freesasa.structureFromBioPDB(strucChain1)
    fs_chain2 = freesasa.structureFromBioPDB(strucChain2)
    t2 = time.perf_counter()

    area_pp = freesasa.calc(fs_pp).totalArea()
    area_chain1 = freesasa.calc(fs_chain1).totalArea()
    area_chain2 = freesasa.calc(fs_chain2).totalArea()
    tf = time.perf_counter()

    buried_area = (area_chain1 + area_chain2 - area_pp)
    tf = time.perf_counter()
    logger.debug(f"Sasa values: Chain 1 = {round(area_chain1, 3)}, Chain 2 = {round(area_chain2, 3)}, Total = {round(area_pp, 3)}")
    logger.debug(f"Runtime calculate_buried_area: {round((tf-ti)*1000, 1)}ms ({round((t1-ti)*1000, 1)}ms model buiilding, {round((t2-t1)*1000, 1)}ms loading, {round((tf-t2)*1000, 1)}ms sasa calc)")
    return round(buried_area, 3)

def calculate_buried_area_biotite(atomarray_biotite:struc.AtomArray, chain1:struc.AtomArray, chain2:struc.AtomArray, probe_radius:float=1.4):
    """
        Calculates the buried surface area using biotite which is defined as surface area of the two chains
        subtracted from the surface area of the complex.
    """
    ti = time.perf_counter()

    sasa12 = np.sum([s for s in struc.sasa(atomarray_biotite, probe_radius=probe_radius) if math.isfinite(s)])
    sasa1 = np.sum([s for s in struc.sasa(chain1, probe_radius=probe_radius) if math.isfinite(s)])
    sasa2 = np.sum([s for s in struc.sasa(chain2, probe_radius=probe_radius) if math.isfinite(s)])

    logger.debug(f"Sasa values: Chain 1 = {round(sasa1, 3)}, Chain 2 = {round(sasa2, 3)}, Total = {round(sasa12, 3)}")
    buried_area = (sasa1 + sasa2 - sasa12)
    tf = time.perf_counter()
    logger.debug(f"Runtime calculate_buried_area: {round((tf-ti)*1000, 1)}ms ")
    return round(buried_area, 3)


def calculate_min_distance(atomarray_biotite:struc.AtomArray, cutoff:float=5.0, max_cutoff:float = 15.0):
    """
        Calculates the minimum distance [Angstrom] between the two chains of a protein complex using biotite.
        The minimum distance is defined as the distance between the backbone (CA atoms) if is subceeds
        the cutoff value. For distances above cutoff the algorithm reports NaN

        You may whish to apply the cutoff value not for the backbone only but for all atoms of the residue.
        For this, set the max_cutoff [Angstrom] to something above cutoff (for example twice) and this function
        will report distances above cutoff if a) at least one pair of atoms in the two residues has a distance
        below cutoff and b) the backbone distance is still below max_cutoff.
        This will require MUCH more computational power and should therefore only be enabled if necessary.
    """
    ti = time.perf_counter()
    chains = list(set(struc.get_chains(atomarray_biotite)))
    assert len(chains) == 2

    chain1 = atomarray_biotite[atomarray_biotite.chain_id == chains[0]]
    chain2 = atomarray_biotite[atomarray_biotite.chain_id == chains[1]]

    chain1_backbone = chain1[chain1.atom_name == "CA"]
    chain2_backbone = chain2[chain2.atom_name == "CA"]
    
    min_distance = float("inf")
    t1 = time.perf_counter()

    # max_cutoff is implemented to mimic the same behaviour as the ISS code which used pymol.

    for ca1 in chain1_backbone:
        for ca2 in chain2_backbone:
            if (dist := struc.distance(ca1, ca2)) < cutoff:
                min_distance = min(min_distance, dist)
                continue
            elif dist <= max_cutoff and dist < min_distance: # If max_cutoff is set, check the individual atoms
                for a1 in chain1[chain1.res_id == ca1.res_id]:
                    for a2 in chain2[chain2.res_id == ca2.res_id]:
                        if struc.distance(a1, a2) <= cutoff:
                            break
                    else: # Runs after loop finished normally
                        continue
                    break # This only runs if there is a break in the inner loop because of previous continue statement
                else:
                    # Only calculate min_distance if there is the atom wise distance of the residues is below cutoff
                    continue
                min_distance = min(min_distance, dist)

    tf = time.perf_counter()
    logger.debug(f"Runtime calculate_min_distance: {round((tf-ti)*1000, 1)}ms ({round((t1-ti)*1000, 1)}ms generating chains, {round((tf-t1)*1000, 1)}ms calculating distance)")
    
    return round(float(min_distance), 3) if math.isfinite(min_distance) else float('NaN')

def calculate_hbonds(atomarray_biotite:struc.AtomArray):
    """
        Calculates the number of hbonds between two chains of a protein complex using biotites AtomArray
    """
    ti = time.perf_counter()
    chains = list(set(struc.get_chains(atomarray_biotite)))
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


def calculate_disulfide_bonds(structure_biopy:BioPy_PDBStructure, cutoff:float=2.1):
    """
        Calculates the number of disulfide bonds between the two chains of a protein complex using biopython.
        For this, sulfer atoms in cysteins with a distance less than 2.1 Angstrom are searched
    """
    ti = time.perf_counter()
    chains = [c for c in structure_biopy.get_chains()]
    assert len(chains) == 2

    chain1 = structure_biopy[0][chains[0].id]
    chain2 = structure_biopy[0][chains[1].id]

    disulfide_bonds = 0
    for res1 in [r for r in chain1 if r.resname == "CYS"]:
        for res2 in [r for r in chain2 if r.resname == "CYS"]:
            for atom1 in [a for a in res1 if a.id == "SG"]:
                for atom2 in [a for a in res2 if a.id == "SG"]:
                    distance = atom1 - atom2
                    if distance <= cutoff:
                        disulfide_bonds += 1
                        break # Only one disulfide bond per pair
                else: # This three lines pass the break from the inner to the outer loop
                    continue
                break
    tf = time.perf_counter()
    logger.debug(f"Runtime calculate_saltbridges: {round((tf-ti)*1000, 1)}ms")
    
    return disulfide_bonds


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

    saltBridges_ac = {"ASP":"a", "GLU":"a", "ARG":"b", "LYS":"b", "HIS": "b"} # a: Acidic, b: Basic
    saltBridges_atoms = ['OD1', 'OD2', # Aspartate
                        'OE1', 'OE2', # Glutamate
                        'NH1', 'NH2', 'NE', # Arginin 
                        'NZ', # Lysin
                        'ND1', 'NE1', 'AE1', 'AE2' # Histidin
                        ]

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
                        break # Only one salt bridge per pair
                else: # This three lines pass the break from the inner to the outer loop
                    continue
                break
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
    if structure_biopy is None or atomarray_biotite is None: raise ProteinStructureWarning(f"The strucuture {structure_name} (file {path.name}) can't be opened")
    if len([c for c in structure_biopy.get_chains()]) != 2: raise ProteinStructureWarning(f"The strucuture {structure_name} (file {path.name}) has invalid chain length")

    buried_area = calculate_buried_area(structure_biopy) if _freesasa_ready else calculate_buried_area_biotite(atomarray_biotite)
    hbonds = calculate_hbonds(atomarray_biotite)
    min_distance = calculate_min_distance(atomarray_biotite)
    salt_bridges = calculate_saltbridges(structure_biopy)
    hydrophobic_interactions = calculate_hydrophobic_interactions(structure_biopy)
    disulfide_bonds = calculate_disulfide_bonds(structure_biopy)

    tf = time.perf_counter()
    logger.log(level=LOGLEVEL_ADDITIONAL_INFO, msg=f"parsed {structure_name} (file {file_name}) in {round((tf-ti), 3)}s")
    return {
        'structure_name': structure_name,
        'file': file_name,
        'hbonds': hbonds,
        'salt_bridges': salt_bridges,
        'buried_area': buried_area,
        'min_distance': min_distance,
        'hydrophobic_interactions': hydrophobic_interactions,
        "disulfide_bonds": disulfide_bonds,
    }

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
    logger.info(f"Found {structures_count} structures")
    return pathObj


# This function is called from the main processes with a pathObj
def _run_task(args) -> tuple[dict|None, str]:
    """
        Helper function called from the main process using multiprocessing.Pool and pool.imap_unordered.
        Returns a tuple. The first value is either or a dict containing the measurement parameters or None on a error, while the second value
        is the output of the logging function.
    """
    path, structure_name = args
    try:
        r = EvaluateStructure(path, structure_name)
    except ProteinStructureWarning as ex:
        logger.warning(str(ex))
        r = None
    output = stdout.getvalue().strip("\n")
    return (r, output)

def Run(pathObj: list[tuple[pathlib.Path, str]], num_threads=None) -> pd.DataFrame|None:
    """
        Measures the given paths and returns the result as pandas Dataframe.
        pathObj is a list of tuples of (path_to_pdf: pathlib.Path, structure_name: str)
        [The structurename is used for the output as filenames often are not unique]
    """

    if len(pathObj) == 0:
        logger.warning(f"You provided an empty pathObj")
        return
    logger.info(f"Started Taskpool of {num_threads} processes for {len(pathObj)} files")
    t0 = time.perf_counter()
    with Pool(initializer=_WorkerThreadIni, initargs=[logger.level], processes=(num_threads if num_threads is not None else cpu_count())) as p:
        results = []
        _ti = t0
        _ti_n = 0
        for i, rmsg in enumerate(p.imap_unordered(_run_task, pathObj)):
            r, output = rmsg
            if len(output) > 0:
                print(output)
            _ti_n += 1
            if time.perf_counter() - _ti > 5:
                _speed = ((time.perf_counter() - _ti)/_ti_n)**-1 if _ti_n > 0 else 0
                _speed_avg = ((time.perf_counter() - t0)/i)**-1 if i > 0 else 0
                _eta = (len(pathObj) - i)/_speed_avg if _speed_avg != 0 else -1
                _ti = time.perf_counter()
                _ti_n = 0
                logger.info(f"{int(100*i/len(pathObj))}% - ETA {str(datetime.timedelta(seconds=int(_eta))) if _eta >= 0 else '?'} | current speed {round(_speed, 3)} s⁻¹ | average speed {round(_speed_avg, 3)} s⁻¹")
            if r is not None:
                results.append(r)
    _speed_avg = ((time.perf_counter() - t0)/len(pathObj))**-1
    num_errors = len(pathObj) - len(results)
    logger.info(f"Finished processing {len(pathObj)} objects{f' ({num_errors} objects produced an error)' if num_errors > 0 else ''} in {str(datetime.timedelta(seconds=int(time.perf_counter() - t0)))} | average speed {round(_speed_avg, 3)} s⁻¹")
    if len(results) == 0:
        return None
    return pd.DataFrame(results).sort_values(["structure_name", "file"])