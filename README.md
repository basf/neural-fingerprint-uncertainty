# Analysis of Uncertainty of Neural Fingerprint-based Models
This repository contains the code to reproduce the results of the paper "Analysis of Uncertainty of Neural Fingerprint-based Models" (under review).
## Abstract
Estimating the uncertainty of a model prediction is crucial for the deployment of machine learning models in nearly all applications of cheminformatics.
For many standard machine learning models, uncertainty estimates are well studied, however, their predictive performances are often inferior to those of graph neural networks (GNNs).
In this study, we investigate whether the neural fingerprint extracted from a GNN can be used to improve the uncertainty estimates of classical machine learning models.
## Reproducing the results
### DVC
The experiments are managed using [DVC](https://dvc.org/), where each step is specified in the [`dvc.yaml`](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#dvcyaml) file.
Running the pipeline will create a dvc.lock file, which contains the hashes of the scripts, input files, and output files, ensuring that the results originate from the provided code and data.
Typically, the cache is stored in a remote, however, as no suitable remote is available, the cache is stored in a tarball, which requires manual extraction.
The following sections describe how to set up the project and reproduce the results.
### Commands to reproduce the results
1. Clone the repository
```bash
git clone https://github.com/basf/neural-fingerprint-uncertainty.git
cd neural-fingerprint-uncertainty
```
2. Install the requirements
```bash
pip install -r requirements.txt
```
3. Unzip the dvc cache
```bash
tar -xf dvc_cache.tar.gz .dvc/
```
4. Pull the data
```bash
dvc pull
```
5. Reproduce the results
```bash
dvc repro
```
## Workflow of the experiments
### Molecular standardization
The molecular standardization is performed using [molpipeline](https://github.com/basf/molpipeline).
Details of the standardization are provided in the [01_preprocess_data.py](scripts%2F01_preprocess_data.py) script.
### Creating the folds
The data is split into 5 folds using the `StratifiedKFold` method and the `GroupKFold` method, where the group is determined by Agglomerative Clustering.
The details of the fold creation are provided in the [02_assign_groups.py](scripts%2F02_assign_groups.py) script.
### ML experiments with Morgan fingerprints
The Morgan fingerprints are used to train the classical machine learning models.
The details of the experiments are provided in the [03_ml_experiments.py](scripts%2F03_ml_experiments.py) script.
### ML experiments with neural fingerprints
The neural fingerprints are extracted from a pre-trained [Chemprop](https://github.com/chemprop/chemprop) model.
In addition to the neural fingerprints, the GNN is also used to predict the target values.
The details of the experiments are provided in the [04_neural_fingerprint_predictions.py](scripts%2F04_neural_fingerprint_predictions.py) script.
### Create plots for each endpoint
The results of the experiments are visualized using matplotlib, where the plots are saved in the `[figures](data%2Ffigures)` folder.
Figures have to be loaded using the commands provided [above](#Commands to reproduce the results).
Code for the plots is provided in the [05_create_plots.py](scripts%2F05_create_plots.py) script.
### Plots used in the paper
The plots used in the paper are saved in the [final_figures](data%2Ffigures%2Ffinal_figures) folder.
The code for the plots is provided in the [06_create_final_figures.py](scripts%2F06_create_final_figures.py) script.
### Tables used in the paper
The tables used in the paper were logged and directly extracted from the console.
A copy of the console output is provided in the file [07_create_final_tables.log](logs%2F07_create_final_tables.log).

