# Analysis of Uncertainty of Neural Fingerprint-based Models
This repository contains the code to reproduce the results of the paper "Analysis of Uncertainty of Neural Fingerprint-based Models" (under review).
## Abstract
Estimating the uncertainty of a model prediction is crucial for the deployment of machine learning models in nearly all applications of cheminformatics
For many standard machine learning models, uncertainty estimates are well studied, however, their predictive performances are often inferior to those graph neural networks (GNNs).
In this study, we investigate whether the neural fingerprint extracted from a GNN can be used to improve the uncertainty estimates of classical machine learning models.
## Reproducing the results
### DVC
The experiments are managed using [DVC](https://dvc.org/), where each step is specified in a  [`dvc.yaml`](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#dvcyaml) file.
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