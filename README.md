# Multi-Scale and Directional Frequency-Guided Motion Artifact Reduction in Knee MRI

Can explicit multi-scale and directional image representations (gradients, Laplacian pyramid bands, Gabor responses) help a lightweight residual network suppress real MRI motion artifacts while preserving genuine high-frequency anatomy?

- Dataset: [KMAR-50K](https://doi.org/10.1038/s41597-025-05439-1) — paired motion-corrupted / rescanned knee MRI (Xi et al., Sci Data 2025). Download: Mendeley [part 1](https://data.mendeley.com/datasets/xw7mrg7ntg), [part 2](https://data.mendeley.com/datasets/95w9f5tzz8).

## Layout
```
configs/   experiment YAMLs
data/      raw/ + processed/ (git-ignored), inventory.csv, splits.json
dipmri/    library: data, features, models, classical, metrics, train
scripts/   00_inventory → 01_preprocess → 10_analysis → 20_classical → 30_train → 40_evaluate → 50_figures
runs/      checkpoints/logs (git-ignored)
results/   analysis/, tables/, figures/
```

## Setup
```bash
python -m venv /archive/simran.singh/envs/simran_kmar50
source /archive/simran.singh/envs/simran_kmar50/bin/activate
pip install -r requirements.txt
```
