# Plan: Frequency-Guided Motion Artifact Reduction in Knee MRI (DIP Project)

## Context
This is the DIP course project for Simran Singh Sandral and Aashish Anand. The proposal asks one question: **do explicit DIP representations (gradient G, multi-scale frequency H, directional D) help a lightweight residual network remove real motion artifacts from knee MRI, compared with classical filters and an image-only U-Net, without blurring real anatomical edges?**

`/archive/simran.singh/DIP_Project` exists and is empty. The dataset is not downloaded yet. GPU 1 (H100, 95 GB) is currently free. The other three are nearly full.

When this plan is approved, the first action is to write it to `/archive/simran.singh/DIP_Project/PLAN.md`, together with a short `CLAUDE.md` (project summary, directory layout, how to run) so later sessions can pick up from there. After that, work goes phase by phase, and the user checks each phase before the next one starts.

---

## 1. Dataset: KMAR-50K (Xi et al., Sci Data 2025, doi 10.1038/s41597-025-05439-1)
- 1,190 patients, 1,444 paired sequences, about 62.5k slices. The data is paired NIfTI files (one with motion, one rescanned without motion) that are already bias-field corrected, min-max normalized and registered pairwise.
- Download: Mendeley `xw7mrg7ntg` (part 1, v6) + `95w9f5tzz8` (part 2) → `DIP_Project/data/raw/` (each part gets its own empty subdirectory; read it as data only).
- **Step 1.1 Inventory** (`scripts/00_inventory.py`): parse the file names and NIfTI headers into `data/inventory.csv`. Columns: patient, sequence, plane, shape, spacing, and corrupted/clean path.
- **Step 1.2 Pick the subset**: one sequence and one plane, as the proposal says. Choose the (sequence, plane) with the most pairs. The paper's U-Net benchmark is on the transverse plane (PSNR 28.47 / SSIM 0.927), which gives a sanity anchor if we use that plane. Decide after looking at the inventory.
- **Step 1.3 Pair quality check**: real rescans are never perfectly aligned. For each pair, compute the NCC and mutual information between X and Y. Flag pairs below a threshold and report how many are dropped, because misregistration lowers every method's PSNR.
- **Preprocessing** (`dipmri/data.py`):
  - load each volume and take 2D slices along the chosen plane;
  - normalize with a per-volume 99.5th percentile (re-checked even though the data is already min-max normalized);
  - center-crop or pad to 256×256, resizing only if the in-plane size is very different;
  - drop near-empty slices (foreground fraction under about 5% from an Otsu mask on Y);
  - cache to `data/processed/*.npz` (float16) with a foreground mask per slice.
- **Splits**: 70/15/15 split **by patient**, fixed seed, saved to `data/splits.json` and never regenerated. The split code asserts that no patient appears in two splits.
- **Augmentation** (train only): horizontal flip and ±5° rotation, applied identically to X and Y. **No vertical flip or 90° rotation**, because ghosting runs along the phase-encode direction and those transforms would destroy that directional signal.

## 2. Phase A: Paired spatial-frequency analysis (the DIP core, contribution 1–2)
`scripts/10_analysis.py` → `results/analysis/` (figures + CSV). This runs on the train split only, so nothing about the test set feeds design choices.

For each pair, the script analyzes X, Y and the **residual R = X − Y** (R is the artifact itself, so it is the most useful thing to look at):
1. **Fourier**:
   - mean log|F_X| − log|F_Y| map (ΔF from Eq. 5);
   - mean |F_R|²;
   - expected finding: energy concentrated along the phase-encode axis (ghost replicas).
2. **Radial energy profile** E(r) for X, Y and R, plus the ratio E_X/E_Y per radial band (low, mid, high).
3. **Angular energy**: spectrum energy in 12 angular sectors. This measures anisotropy, i.e. whether motion corruption is direction-dependent.
4. **Gradients**: Sobel magnitude and orientation histograms for X vs Y, plus total variation.
5. **Multi-scale**: Laplacian pyramid (4 levels) and wavelet (db2, 3 levels; LH/HL/HH) subband energy for X vs Y vs R.
6. **Directional**: Gabor bank (4–8 orientations × 2 scales) mean response for X vs Y, as a polar plot.
7. **Statistics**:
   - paired Wilcoxon test per feature, aggregated per patient so slices don't inflate n;
   - Spearman correlation between each feature and artifact severity, defined as the per-slice NRMSE(X, Y).

**Output**: a list of which bands and orientations change reproducibly. This list decides the exact H and D configuration (for example, which Gabor orientations and which pyramid levels), and it also motivates the directional classical filter.

## 3. DIP representations (inputs Z = [X, G, H, D])
`dipmri/features.py`. All three are **fixed (non-learnable) torch convolutions computed on the GPU on the fly**, so augmentation stays consistent and nothing needs to be cached.

| Block | Channels | Definition |
|---|---|---|
| G(X) | 3 | Sobel gx, gy, magnitude |
| H(X) | 4 | Laplacian pyramid bands L0–L2 upsampled to 256² + low-pass residual (the Eq. 6 {L, H1..Hk}) |
| D(X) | 8 (default) | Gabor magnitude, 4 orientations × 2 scales (adjusted after Phase A) |

Each channel is normalized with a per-channel mean and std computed on the training set.

## 4. Models (`dipmri/models.py`)
Every learned model uses the same backbone so the comparison is controlled:
- **Backbone**: a 4-level U-Net with base width 32, about 7.8M parameters. Each block is Conv-GroupNorm-SiLU ×2. Downsampling uses strided conv and upsampling uses bilinear + conv.
- **M1 U-Net (direct)**: input X, output Ŷ.
- **M2 U-Net (residual)**: input X, output ΔX, with Ŷ = X + ΔX. This is the proposal's control for "the gain comes only from residual learning".
- **M3 FG-ResUNet (proposed)**: input Z (1+3+4+8 = 16 channels), residual output. Only the first conv differs from M2, so the parameter difference is under 0.1%.
- **M3-dual (optional, only if time allows)**: a separate small feature stem fused into the encoder through FiLM or concatenation at each scale. Treat it as an extension, not the main result.

**Training**:
- loss: L1 + 0.2·(1 − SSIM);
- optimizer: AdamW, lr 2e-4 with cosine decay;
- batch size 16, bf16 AMP;
- about 100 epochs, or a fixed iteration budget that is the same for all models;
- checkpoint selection by validation PSNR;
- **3 seeds** for M1, M2 and M3.

Runs are driven by YAML configs in `configs/`, and each run writes to `runs/<name>/`.

## 5. Baselines
**Classical** (`dipmri/classical.py`). Each one's hyperparameters are grid-searched on **val** for PSNR, then run once on test:
- Gaussian (σ)
- Median (k)
- Wiener (`scipy.signal.wiener`, window)
- Frequency-domain:
  - (a) Butterworth low-pass (cutoff, order);
  - (b) a **directional/anisotropic filter along the phase-encode axis** chosen from the Phase A findings. This is the "smart DIP" baseline.
- Wavelet denoising (`skimage.restoration.denoise_wavelet`, BayesShrink/VisuShrink)
- Optionally non-local means

**Learned**: M1 and M2 as above.

## 6. Ablations
All ablations are residual, use the same backbone and budget, and run with 1 seed (3 seeds if a difference looks borderline).
1. Single blocks: X+G, X+H, X+D
2. Pairs: X+G+H, X+G+D, X+H+D
3. Full set: X+G+H+D (M3)
4. **Channel-count control**: X + 15 fixed *random* filters. This tests whether the gain comes from the DIP content or just from having more input channels.
5. **Learnable-stem control**: M2 with an extra learnable 16-channel conv stem. This tests whether the network can learn the same filters on its own.
6. Optional: adding an FFT-magnitude L1 loss term, applied to M2 and M3.

## 7. Evaluation (`dipmri/metrics.py`, `scripts/40_evaluate.py`)
Metrics are computed on the test split, per slice, then aggregated per patient. Each one is reported both full-image and inside the foreground mask.
- **Pixel**: PSNR, SSIM, MAE, NRMSE.
- **Structure**:
  - HFEN (LoG high-frequency error norm, a standard MRI metric);
  - gradient-magnitude similarity (GMSD);
  - edge F1 (Canny on Ŷ vs Y).
- **Frequency**:
  - radial-spectrum error per band (low/mid/high);
  - wavelet-subband reconstruction error;
  - angular-energy error along the phase-encode direction.
- **Stratified**: results by artifact-severity tercile (mild / moderate / severe input NRMSE).
- **Statistics**: paired Wilcoxon test (M3 vs M2, M3 vs best classical) over test patients, plus bootstrap 95% CIs.
- **Qualitative**: a fixed panel of about 8 test slices showing X, the classical baseline, M2, M3, Y and the |error| maps, with zoomed crops on cartilage/meniscus edges.
- Sanity check: on the transverse plane, M1 should land roughly near the paper's PSNR of about 28.5. If it falls far below, check preprocessing.

## 8. Directory layout (`/archive/simran.singh/DIP_Project`)
```
PLAN.md  CLAUDE.md  README.md  requirements.txt
configs/            # yaml per experiment
data/raw/ data/processed/ data/inventory.csv data/splits.json
dipmri/             # data.py features.py models.py classical.py metrics.py train.py utils.py
scripts/            # 00_inventory, 01_preprocess, 10_analysis, 20_classical, 30_train, 40_evaluate, 50_figures
runs/               # checkpoints + logs per experiment
results/            # analysis/, tables/, figures/
notebooks/          # optional exploration
```
**Environment**: a new venv at `/archive/simran.singh/envs/dip`. The conda install is broken (see memory), so don't use it. Packages: torch, nibabel, numpy, scipy, scikit-image, PyWavelets, pandas, matplotlib, pyyaml, tqdm. Training runs on GPU 1 (`CUDA_VISIBLE_DEVICES=1`), after checking it is still free.

## 9. Execution order / milestones
1. **Setup**: create the venv, download the data, run the inventory, choose the subset, run the pair-quality check, preprocess, make the splits. *(User check: subset choice and dropped-pair count.)*
2. **Phase A analysis**: produce the figures and stats table, then fix the H/D configuration. *(User check.)*
3. **Classical baselines**: tune on val, evaluate on test, produce the first results table.
4. **M1, M2**: train them, then confirm the sanity PSNR.
5. **M3**, then the ablations and controls.
6. **Final evaluation**: stratified results, significance tests, figures, results tables for the report.

## 10. Verification
- **Splits**: a unit assert that the train/val/test patient sets don't overlap.
- **Features**: on a synthetic image (a disk plus oriented stripes), the Gabor responses peak at the expected orientation, and the Laplacian pyramid bands sum back to the input (error under 1e-5).
- **Metrics**: PSNR and SSIM are checked against `skimage.metrics` on random pairs. Identity and X-as-prediction give the expected values (the "uncorrected" row).
- **Training smoke test**: each model overfits about 16 slices to more than 40 dB in a few hundred steps before the full runs.
- **Reproducibility**: each run saves its config, seed, git hash (after `git init`), and the val/test metric JSON.
