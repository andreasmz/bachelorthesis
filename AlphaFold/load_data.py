""" Load the benchmark set metadata into a pandas table """
# File created by Andreas on 02.06.2025, but based on code from the former "2025-05-10 plots thesis 1.ipynb" notebook

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def dataAF2(path: Path|None = None, more_columns: bool = False) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).parent / "AF2" / "AF2_metrics.tsv"
    dataAF2 = pd.read_csv(path, sep="\t")
    for c in ["chainA_start", "chainA_end", "chainB_start", "chainB_end", "num_mutations", "num_align_atoms_domain", "num_align_resi_domain", "hbonds", "salt_bridges", "hydrophobic_interactions"]:
        if c not in dataAF2.columns:
            print(f"Column {bcolors.FAIL}{c}{bcolors.ENDC} not (yet) in data frame")
            continue
        dataAF2[c] = dataAF2[c].astype(pd.Int64Dtype())
    dataAF2["score_rank"] = dataAF2.groupby("prediction_name")["model_confidence"].rank("first", ascending=False).astype(pd.Int64Dtype())
    dataAF2["RMSD_rank"] = dataAF2.groupby("prediction_name")["RMSD_all_atom"].rank("first").astype(pd.Int64Dtype())
    dataAF2["RMSD_peptide_rank"] = dataAF2.groupby("prediction_name")["RMSD_all_atom_peptide"].rank("first").astype(pd.Int64Dtype())
    dataAF2["DockQ_rank"] = dataAF2.groupby("prediction_name")["DockQ"].rank("first").astype(pd.Int64Dtype())
    dataAF2["intf_avg_plddt_rank"] = dataAF2.groupby("prediction_name")["intf_avg_plddt"].rank("first").astype(pd.Int64Dtype())
    dataAF2["chainA_intf_avg_plddt"] = dataAF2.groupby("prediction_name")["chainA_intf_avg_plddt"].rank("first").astype(pd.Int64Dtype())
    dataAF2["chainB_intf_avg_plddt_rank"] = dataAF2.groupby("prediction_name")["chainB_intf_avg_plddt"].rank("first").astype(pd.Int64Dtype())
    dataAF2["pDockQ_rank"] = dataAF2.groupby("prediction_name")["pDockQ"].rank("first").astype(pd.Int64Dtype())
    dataAF2["iPAE_rank"] = dataAF2.groupby("prediction_name")["iPAE"].rank("first", ascending=False).astype(pd.Int64Dtype())
    return dataAF2


def dataAF3(path: Path|None = None, more_columns: bool = False) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).parent / "AF3" / "AF3_metrics.tsv"
    dataAF3 = pd.read_csv(path, sep="\t")
    for c in ["chainA_start", "chainA_end", "chainB_start", "chainB_end", "num_mutations", "num_align_atoms_domain", "num_align_resi_domain", "hbonds", "salt_bridges", "hydrophobic_interactions"]:
        if c not in dataAF3.columns:
            print(f"Column {bcolors.FAIL}{c}{bcolors.ENDC} not (yet) in data frame")
            continue
        dataAF3[c] = dataAF3[c].astype(pd.Int64Dtype())
    dataAF3["score_rank"] = dataAF3.groupby("prediction_name")["ranking_score"].rank("first", ascending=False).astype(pd.Int64Dtype())
    dataAF3["RMSD_rank"] = dataAF3.groupby("prediction_name")["RMSD_all_atom"].rank("first").astype(pd.Int64Dtype())
    dataAF3["RMSD_peptide_rank"] = dataAF3.groupby("prediction_name")["RMSD_all_atom_peptide"].rank("first").astype(pd.Int64Dtype())
    dataAF3["DockQ_rank"] = dataAF3.groupby("prediction_name")["DockQ"].rank("first").astype(pd.Int64Dtype())
    dataAF3["intf_avg_plddt_rank"] = dataAF3.groupby("prediction_name")["intf_avg_plddt"].rank("first").astype(pd.Int64Dtype())
    dataAF3["chainA_intf_avg_plddt"] = dataAF3.groupby("prediction_name")["chainA_intf_avg_plddt"].rank("first").astype(pd.Int64Dtype())
    dataAF3["chainB_intf_avg_plddt_rank"] = dataAF3.groupby("prediction_name")["chainB_intf_avg_plddt"].rank("first").astype(pd.Int64Dtype())
    dataAF3["pDockQ_rank"] = dataAF3.groupby("prediction_name")["pDockQ"].rank("first").astype(pd.Int64Dtype())
    dataAF3["iPAE_rank"] = dataAF3.groupby("prediction_name")["iPAE"].rank("first", ascending=False).astype(pd.Int64Dtype())
    return dataAF3


def dataSolved(path: Path|None = None, hydrogens: bool = True) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).parent / "experimentally solved" / f"solved{'_hydrogens' if hydrogens else ''}_metrics.tsv"
    dataSolved = pd.read_csv(path, sep="\t")
    return dataSolved

def dataAF(dataAF2: pd.DataFrame, dataAF3: pd.DataFrame) -> pd.DataFrame:
    dataAF = pd.merge(
        left=dataAF3,
        right=dataAF2,
        left_on=["benchmark_set", "prediction_name", "model_id"],
        right_on=["benchmark_set", "prediction_name", "model_id"],
        suffixes=["_AF3", "_AF2"],
        how="inner"
    )
    for c in ["chainA_length", "chainB_length", "chainA_id", "chainB_id", "chainA_start", "chainA_end", "chainB_start", "chainB_end", "PDB_id", "ELM_instance", "DDI_pfam_id", "PDB_id_random_paired", "ELM_instance_random_paired", "DDI_pfam_id_random_paired", "sequence_initial", "sequence_mutated", "num_mutations", "score_rank"]:
        if len(dataAF[~(dataAF[c+"_AF2"] == dataAF[c+"_AF3"]) & (~dataAF[c+"_AF2"].isna()) & (~dataAF[c+"_AF3"].isna())]) > 0:
            print(f"Unmatched column {c}")
            continue
        dataAF.drop(columns=[c+"_AF2"], inplace=True)
        dataAF.rename(columns={c+"_AF3": c}, inplace=True)
    dataAF.rename(columns={"ranking_score": "ranking_score_AF3"}, inplace=True)
    dataAF.rename(columns={"model_confidence": "model_confidence_AF2"}, inplace=True)
    dataAF.rename(columns={"ipSAE": "ipSAE_AF3"}, inplace=True)

    return dataAF