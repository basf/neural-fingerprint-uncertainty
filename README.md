# Analysis of Uncertainty of Neural Fingerprint-based Models
## How to reproduce the results
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