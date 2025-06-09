# Bachelorthesis Molecular Biology at AG Luck (IMB Mainz)

Topic: Evaluating AlphaFold version 2 and 3 for their ability to accurately predict protein complex structures

### /AlphaFold

In this folder you find the benchmark set as well as all final evaluations used to gather data presented in the thesis. The important file is `Benchmark set.ipynb` containg statistics on the benchmark set. Besides that, the `AF metrics.ipynb` notebook contains the code used to add most of the metrics (all other metrics were added by the IMB server). In case you want to run a bulk of AF3 predictions on the IMB server, `AF3 raw output parsing.ipynb` may be useful. Last, `load_data.py` is an internal script to load the benchmark set from a given path.

Besides that, the folder contains the metrics for all AlphaFold 2 and 3 runs including the new interaction interface metrics. They are stored in a tsv table as well as an excel sheet. Also, the experimentally solved structures used as reference as well as their interface metrics are included.

### /Interface metrics

Here you can find the developed interface interaction library intendend to measure chemical and physical properties in the interface of PPIs (`measure_PPI.py`). Also, you can find here the `add_hydrogens.ipynb` notebook used to add hydrogens to the experimentally solved structures and AF3 predictions.

### /Plots

Here all plots of the bachelor thesis as well as the notebooks used to generate them can be found

### /dev

This dev folder should be understand as a kind of lab notebook, as it contains my experiments on the dataset. That is the reason why the notebooks are all named with the date I created them, as they do not contain final code (final code is copied to the /AlphaFold, /Interface metrics or /Plots folder). The notebooks consists mostly of try-and-error code to find ways how to deal with the data.

### /external

Here external code like the ipSAE metric script or code from other group members (e.g. from the Lee et al. (2024) paper) are stored.