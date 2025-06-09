# AlphaFold 3 predictions

Here the metadata including all metrics for the AlphaFold 3 runs can be found with and without added hydrogens:

* <a href="AF3_metrics.tsv">AF3_metrics.tsv</a>
* <a href="AF3_metrics.xlsx">AF3_metrics.xlsx</a>
* <a href="AF3_hydrogens_metrics.tsv">AF3_hydrogens_metrics.tsv</a>
* <a href="AF3_hydrogens_metrics.xlsx">AF3_hydrogens_metrics.xlsx</a>

The code used to generate this files can be found in the <a href="../AF metrics.ipynb">AF metrics.ipynb</a> notebook. The documentation on the columns can be found in the <a href="../Benchmark set columns.xlsx">Benchmark set columns.xlsx</a> file.

### Prediction files

The files themself are to big for github and can be found in the group drive.

### Raw server output

* <a href="AF3_output.tsv">AF3_output.tsv</a>
* <a href="AF3_output.xlsx">AF3_output.xlsx</a>

are merged tables containing the metrics calculated by the IMB server (e.g. ranking score, chain length, pDockQ, ...), but not the template dependent metrics (e.g. ipSAE, RMSD, Interface Interaction metrics, ...). It was created using the <a href="../AF3 raw output parsing.ipynb">AF3 raw output parsing.ipynb</a> notebook.