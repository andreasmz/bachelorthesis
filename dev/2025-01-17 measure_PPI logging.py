import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
import time

libpath = Path("./src").resolve()
sys.path.insert(0, str(libpath))
import measure_PPI

if __name__ == "__main__":
    AF_prediction_results = Path("./ressources/AF_predictions/AF_prediction_randomized_DMI_results.xlsx").resolve()
    AF_prediction_metrics = Path("./ressources/AF_predictions/AF_metrics_all_structures.tsv").resolve()
    AF_DMI_structures_folders = [Path("./ressources/AF_DMI_structures").resolve() / p for p in ['AF_DMI_structures1', 'AF_DMI_structures2', 'AF_DMI_structures3', "AF_DMI_mutated_structures"]]
    AF_DDI_structures_path = Path("./ressources/AF_DDI_structures").resolve()
    solved_DMI_structures_path = Path("./ressources/DMI_solved_structures_hydrogens").resolve()
    solved_DDI_structures_path = Path("./ressources/DDI_solved_structures_hydrogens").resolve()

    for p in [AF_prediction_results, AF_prediction_metrics, AF_DDI_structures_path, solved_DMI_structures_path, solved_DDI_structures_path] + AF_DMI_structures_folders:
        if not p.exists():
            print(f"{p} does not point to a valid path")

    pathObj = {}
    measure_PPI.WalkFolder(solved_DMI_structures_path, pathObj=pathObj)
    measure_solved_DMI = measure_PPI.Run(list(pathObj.values())[0:1], num_threads=1)

    