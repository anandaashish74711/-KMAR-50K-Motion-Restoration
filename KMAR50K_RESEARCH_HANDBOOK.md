# KMAR-50K Knee MRI Motion Artifact Restoration — Research & Engineering Handbook

**Project:** Multi-Scale and Directional Frequency-Guided Motion Artifact Reduction in Knee MRI  
**Documentation snapshot:** 10 November 2026  
**Status:** Core engineering implemented; preliminary controlled experiments completed; matched full-epoch frequency experiment and rigorous research validation pending  
**Repository:**  https://github.com/anandaashish74711/-KMAR-50K-Motion-Restoration
**Default branch observed:** `origin/main`  
**Feature branch:** `feature/kmar50k-motion-restoration`.

Sequence:

1. `/home2/aashish.a/kmar50k_code` was not initially a Git repository.
2. Initialized Git and configured `origin` to the target HTTPS repository.
3. `git fetch origin` succeeded and discovered `origin/main`.
4. Initial `git switch -c ... origin/main` was blocked by an existing untracked `.gitignore`; the local file was backed up and its ignore rules appended after switching.
5. Created `feature/kmar50k-motion-restoration` from `origin/main`.
6. Staged the source scripts, model/data modules, and `.gitignore`.
7. Unstaged `manifests/dataset_manifest.csv` because it contained cluster-specific absolute paths and potentially sensitive identifiers; added it to `.gitignore`.
8. Excluded raw MRI volumes, checkpoints, temporary files, and generated outputs.
9. `git push -u origin feature/kmar50k-motion-restoration` **succeeded**; remote branch created and upstream tracking set. The remote `main` branch was **not changed**.

**Branch URL:** https://github.com/Simran32909/knee-mri-motion-dip/tree/feature/kmar50k-motion-restoration  
**Pull request URL:** https://github.com/Simran32909/knee-mri-motion-dip/pull/new/feature/kmar50k-motion-restoration

**Unresolved:** This document itself has not yet been committed. The repository lacks a portable manifest generator/configurable dataset root, verified dependency lockfile, complete experiment configs, and a clean reproducibility README. Check `git status` before any future commit.  
**Working branch:** `feature/kmar50k-motion-restoration`  
**Scope:** This document records the work completed and decisions discussed through the first successful GitHub push. It is a *historical and technical record*, not a claim that all intended experiments have been run.

> **Evidence key:** **Observed** = directly reported by terminal output or audit. **Implemented** = source file and behavior discussed/verified in our work. **Proposed** = research direction not yet demonstrated. **Needs verification** = code/data detail or scientific claim requiring re-check. This distinction is intentional: it prevents hypotheses from being mistaken for results.

---

## 1. Executive summary

We built a paired image-restoration pipeline for motion-corrupted knee MRI using the KMAR-50K dataset. The main comparison is between (1) a lightweight **residual U-Net**, which learns an image-space correction, and (2) a **frequency-guided residual U-Net**, which augments the image features with Sobel directional gradients and multi-level Haar wavelet features through learned gating.

The dataset pipeline constructs a manifest of paired NIfTI volumes, audits matching shapes and affines, excludes questionable geometry, uses a filename-derived group split, and samples 2D slices for 256×256 restoration. Training uses a combined L1 + SSIM loss. Evaluation reports PSNR and SSIM per volume and paired model differences.

**Crucial empirical finding:** In the initial matched *limited-step* comparison under independent normalization, the frequency-guided model performed slightly but consistently **worse** than the baseline on the 227-volume validation set. This rejects the assumption that the *current feature-fusion implementation* is automatically better; it does **not** disprove all frequency-domain restoration methods. A separate **full one-epoch baseline U-Net** run subsequently produced validation PSNR **23.0687 dB** and SSIM **0.73338**, but there is **no matched full-epoch frequency-guided result yet**.

The project source has been recovered from node-local scratch, copied to `/home2/aashish.a/kmar50k_code`, and pushed to `feature/kmar50k-motion-restoration`. The raw MRI volumes, model weights, and local absolute-path manifest were deliberately excluded from Git.

### Current status at a glance

| Workstream | Status | Caveat |
|---|---|---|
| Data download and paired file discovery | Done | Dataset documentation and exact population metadata should be independently checked |
| Manifest and geometry audit | Done | 213 questionable pairs excluded from active splits |
| Filename-derived group splits | Done | **Not verified patient IDs** |
| Image-pair visual audit | Partial | Ten random examples, not a registration proof |
| Dataset loader and normalization | Implemented | Slice-wise normalization; independent target scaling changes intensity interpretation |
| Residual U-Net | Implemented | Lightweight 2D architecture |
| Frequency-guided residual U-Net | Implemented | Sobel + Haar + learned gate, not true k-space correction |
| Training, volume-shuffle sampler, seed patch | Implemented | Resumable mid-epoch checkpointing still needed |
| Preliminary full validation comparisons | Done | Short training budgets, single seed |
| One full baseline epoch | Done | No matched full frequency epoch |
| Robust ablations, held-out test, clinical evaluation | Pending | Avoid test-set tuning |
| GitHub feature branch | Pushed | Repo is not independently runnable without manifest/data configuration |

---

## 2. Research question, motivation, and hypothesis

### 2.1 Problem

MRI acquisition is relatively slow and sensitive to patient motion. Motion can create ghosting, blur, signal inconsistency, and apparent displacement of anatomical boundaries. For knee MRI, these distortions can obscure cartilage, menisci, ligaments, and other structures relevant to interpretation. The task is **paired restoration**: given a motion-affected image, estimate a corresponding clean/reference image while retaining real anatomy.

Let the observed slice be \(x\), the reference slice be \(y\), and the learned model be \(f_\theta\). We optimize a restoration objective such as

\[
\hat y=f_\theta(x),\qquad \min_\theta\;\mathbb E_{(x,y)}[\mathcal L(\hat y,y)].
\]

A residual implementation instead learns a correction \(r_\theta(x)\):

\[
\hat y=\operatorname{clip}(x+r_\theta(x),0,1).
\]

This gives the model a natural identity solution (output equals input) when correction is unnecessary.

### 2.2 Original hypothesis

> Multi-scale, directional frequency information will improve knee MRI motion-artifact restoration over an image-only residual U-Net.

**Why it seemed plausible:** Motion introduces structured patterns, including directionally oriented ghosting and frequency-dependent changes. Gradients can expose oriented edges; wavelets can separate structure by scale and orientation. A model might use these cues to locate corruption and preserve anatomy.

**Why it is not guaranteed:** Both artifacts and clinically important anatomy produce high-frequency energy. Fixed Sobel/Haar features may emphasize corrupted edges. Ordinary convolutions already learn frequency-sensitive filters. A simplistic gate may introduce noise rather than useful motion information. Image-domain gradients and wavelets are **not equivalent** to MRI raw k-space measurements.

### 2.3 Hypothesis after preliminary results

**Proposed, not validated:** *Adaptive, motion-aware frequency guidance may improve restoration when it learns to distinguish artifact-related responses from true anatomical structures and selectively controls correction strength.*

A good study must be able to report a negative result honestly. Do not describe the present model as superior unless a matched evaluation supports that claim.

---

## 3. Dataset: what we used and what we know

### 3.1 Source and format

Dataset: **KMAR-50K**, paired motion-affected and corresponding reference knee MRI scans. Earlier project materials describe roughly **1,444 paired sequences from 1,190 patients**, and approximately **62,506 slices**; those are source-level figures, not the count of eligible pairs used by our current loader. Two downloaded archive parts were described as approximately **9.73 GB** and **9.45 GB** compressed (about **19.18 GB** combined). Confirm precise dataset-release metadata, permissions, and any radiologist severity annotations against the dataset publication before using those figures in a paper.

Our implementation reads **NIfTI (`.nii.gz`)** image volumes. Pair paths were assembled from the artifact and ground-truth directory structure. The loader extracts corresponding 2D slices.

### 3.2 Manifest

Original local file: `manifests/dataset_manifest.csv`.

Columns recorded:

```text
group_id,plane,volume_id,artifact_path,ground_truth_path,source,
artifact_shape,ground_truth_shape,voxel_spacing,
same_shape,same_affine,geometry_compatible,split
```

Definitions:

| Column | Meaning |
|---|---|
| `group_id` | Filename-derived grouping key; **not proven a patient identifier** |
| `plane` | Scan plane, e.g., sagittal, coronal, transverse/transection nomenclature |
| `volume_id` | Pair identifier |
| `artifact_path` | Original corrupted NIfTI path |
| `ground_truth_path` | Reference NIfTI path |
| `source` | Source archive/partition |
| `artifact_shape`, `ground_truth_shape` | Volume dimensions |
| `voxel_spacing` | Physical voxel dimensions |
| `same_shape` | Array shape agreement |
| `same_affine` | Affine/geometry agreement under audit criterion |
| `geometry_compatible` | Geometry eligibility flag |
| `split` | `train`, `val`, `test`, or `geometry_review` |

**Important portability limitation:** The original CSV contains absolute paths such as `/scratch/aashish.a/paired_knee_mri_motion/KMAR-50K/...`. It was deliberately **not committed to GitHub**. A portable manifest or path-root rewriting utility remains to be implemented. Do not expose local usernames or path details in public artifacts without review.

### 3.3 Counts and active splits

**Observed:** 1,440 paired volumes were represented in the final working manifest. The active split is:

| Split | Paired volumes | Slices where measured |
|---|---:|---:|
| Train | 899 | 20,773 |
| Validation | 227 | 5,261 |
| Test | 101 | Not recorded in current summary |
| Geometry review / excluded | 213 | Not used for training or main validation |
| **Total** | **1,440** | — |

The 1,440 manifest count and earlier dataset-level 1,444 figure refer to different stages/sources; **the reason for the four-volume difference has not been established here**. Do not silently equate them.

There were **1,189 distinct filename-derived groups** in the working set. The split construction avoided overlap among **active** train/validation/test group IDs. However, this is **not sufficient proof of true patient-level independence**; group names must be checked against authoritative patient metadata. Some excluded geometry-review groups overlap active groups; they were not included in active training/evaluation.

### 3.4 Geometry audit and why it matters

Paired image restoration assumes the target represents the **same anatomy at corresponding coordinates**. A voxel-wise L1 or SSIM objective becomes misleading if images are shifted, differently oriented, or sampled on different grids.

The initial training-pair geometry audit (1,339 training pairs) reported:

| Geometry category | Count |
|---|---:|
| Same shape and affine | 1,126 |
| Same shape, different affine | 125 |
| Different shape and affine | 87 |
| Different shape, same affine | 1 |
| **Total** | **1,339** |

The 101 test pairs were reported geometrically compatible, yielding **1,440** total manifest pairs. The final active selection retained **1,227** train/validation/test pairs and placed **213** into `geometry_review`.

For the 213 review cases, **125** had the same shape but different geometry and **88** had different shapes. Some world-space bounding-box comparisons had no overlap. The maximum reported affine discrepancy in the accepted active set was approximately **0.000346** (under the audit's comparison convention). This is a numeric observation, not a universal physical registration guarantee.

**Decision:** Exclude uncertain geometry rather than train on potentially misregistered supervision. Future work: quantify registration quality in physical coordinates and decide whether carefully validated registration can recover excluded cases without introducing leakage.

### 3.5 Visual pair audit

Ten randomly sampled paired examples were inspected as artifact/reference/difference triptychs. Gross anatomical correspondence appeared plausible. **This does not establish voxel-perfect registration** and cannot rule out localized mismatch. Future audits should use overlays, edge correspondence, affine-aware inspection, and motion-severity stratification.

### 3.6 Data integrity and leakage risks

- Verify every active pair exists and can be loaded; ensure finite values and consistent slice indexing.
- Avoid letting slices from the same true patient appear in different splits.
- Do not use validation or test ground truth to normalize inference inputs.
- Keep `geometry_review` out of model selection until its geometry is resolved.
- Keep the held-out test split untouched while choosing architectures, normalization, or hyperparameters.
- Consider multiple scans per patient when bootstrapping confidence intervals; volume-level samples may not be independent.

---

## 4. Preprocessing: rationale and implementation

### 4.1 Volume loading and 2D slice extraction

Implemented loader: `src/datasets/kmar_dataset.py`, class `KMARDataset`.

Known usage:

```python
KMARDataset(
    manifest_path="manifests/dataset_manifest.csv",
    split="train",
    size=256,
    normalization="independent",
)
```

It filters the manifest by split, enumerates paired slices, loads NIfTI volumes, and returns:

```python
{
    "artifact": Tensor[1, 256, 256],
    "ground_truth": Tensor[1, 256, 256],
    "slice_index": int,
}
```

The loader caches one artifact/reference volume pair to reduce repeated NIfTI reads when sampling adjacent slices. Slices are resized with bilinear interpolation to **256×256**. This is a pragmatic compute choice, not an anatomy-preserving physical resampling to a fixed millimeter spacing. Native physical resolution and aspect ratio can vary, and resize may distort them; record this limitation.

### 4.2 Why MRI intensity normalization is necessary

Unlike CT Hounsfield units, standard MRI image intensities are not intrinsically standardized across scanner settings, coils, protocols, or volumes. A model trained on arbitrary scales may learn intensity conventions rather than artifact removal. We explored percentile normalization using P1 and P99 to reduce sensitivity to outliers.

For an input slice \(x\), a typical percentile transform is:

\[
N(x; a,b)=\operatorname{clip}\left(\frac{x-a}{\max(b-a,\epsilon)},0,1\right),
\]

where \(a=P_1\), \(b=P_{99}\), and \(\epsilon\) prevents division by zero. Confirm the exact epsilon and degenerate-slice handling from `src/datasets/normalization.py` before reproducing implementation-specific values.

### 4.3 Three normalization policies

| Mode | Input scaling | Target scaling | Role |
|---|---|---|---|
| `artifact_only` | P1/P99 from artifact | Same artifact-derived P1/P99 | Earlier baseline; can clip target severely |
| `independent` | Artifact's own P1/P99 | Reference's own P1/P99 | Current main training policy; no target leakage into input |
| `shared_pair` | Joint pair-derived range | Same joint range | Diagnostic only; uses reference information and is **not deployable** |

`src/datasets/normalization.py` implements the modes. The training CLI was updated to accept `artifact_only` and `independent`, excluding `shared_pair` for training. The evaluation script reads normalization from checkpoint metadata to avoid accidentally evaluating a model with the wrong target convention.

**Critical scientific nuance:** Independent normalization is deployment-safe for constructing the input, but target scaling depends on the clean image *during supervised training/evaluation*. The model predicts a normalized reference contrast, not necessarily the original scanner intensity scale. It may hide absolute-intensity discrepancies. PSNR/SSIM across **different normalization policies are not directly comparable** as though they measured exactly the same target task. Compare architectures **within** the same policy.

### 4.4 Audit that motivated normalization changes

`src`/`scripts` audit work checked **1,227 active volumes**, using three representative slices per volume. Reported mean intensity correlation was approximately **0.927569**, mean edge correlation **0.638513**. Under artifact-only normalization, average reference clipping was **0.006863 below** and **0.011272 above** range, but some volumes had extreme clipping:

| Example | Observation under artifact-only normalization |
|---|---|
| `2021_8-XXX_MR1202101060308_sagittal_0.0` | ~79.1% of reference clipped low |
| `2021_105-XX_MR1202104080076_transection_0.0` | ~55% clipped low |
| `2021_258-20210722004267_sagittal_0.0` | ~51% clipped high |
| Another `2021_95...sagittal` example | ~48% combined clipping |

These are **audit observations**, not proof of acquisition corruption or mispairing. They show that a shared artifact-derived intensity range can make a valid clean target appear artificially saturated.

`compare_normalization.py` used **volume-wise** percentiles for diagnostic examples, whereas the actual loader performs **slice-wise** normalization. Keep this difference explicit when interpreting plots and numerical comparisons.

### 4.5 Preprocessing limitations and improvements

- Confirm orientation convention and whether slice axes are consistent across planes.
- Quantify anatomical distortion introduced by square resizing.
- Investigate intensity-standardization schemes that preserve paired contrast relationships while remaining inference-safe.
- Validate all data paths under a configurable dataset root rather than hardcoded scratch paths.
- Test per-volume versus per-slice normalization with identical evaluation definitions.

---

## 5. Data sampling and training efficiency

Implemented: `src/data/samplers.py`, `VolumeShuffleSampler`.

**Problem:** Randomly mixing slices from many volumes can repeatedly decompress/load NIfTI files and defeat the loader's one-volume cache.

**Solution:** Group dataset indices by artifact volume path; shuffle the order of volumes and shuffle slices within each volume. This visits every slice once per full epoch while improving temporal locality of reads. The sampler's pseudocode:

```python
rng = Random(seed + epoch)
shuffle(volume_order)
for volume in volume_order:
    shuffle(indices_for_volume)
    yield each_slice_index
```

`set_epoch(epoch)` changes the shuffle order between epochs. The sampler returns `len(dataset)` as its length. It should be verified for exact coverage and no duplicates in a standalone test, even though its algorithm is structurally sound.

**Observed workload:** 20,773 training slices and 5,261 validation slices. With batch size 2, this corresponds to **10,387 training batches** and **2,631 validation batches** per full epoch (rounding up).

---

## 6. Model A — residual U-Net baseline

**File:** `src/models/unet.py`  
**Constructor:** `UNet(base=16)`  
**Approximate parameter count reported:** **487,009**.

### 6.1 Why U-Net?

U-Net is an encoder–decoder with skip connections. The encoder captures larger-context patterns (useful for diffuse motion artifacts); the decoder reconstructs spatial detail. Skip connections transfer higher-resolution structure that might otherwise be lost through downsampling. A small base width was chosen to fit an **11 GB GTX 1080 Ti** and allow rapid baseline experiments.

### 6.2 Why residual prediction?

The corrupted MRI often contains substantial correct anatomy. Predicting an entire image from scratch can be unnecessarily difficult. The residual model instead predicts **what should change**, and preserves the input where no correction is needed:

\[
\hat y=\operatorname{clip}(x+\Delta_\theta(x),0,1).
\]

The final correction convolution was initialized to zeros, so the network begins approximately as an **identity mapping**. This can stabilize early training and makes the zero-correction behavior explicit.

### 6.3 Risks

- `clip` saturates values outside [0,1] and has zero gradient outside its active interval.
- An identity-biased model can learn small safe corrections while failing to remove severe motion.
- 2D slices do not explicitly enforce consistency across adjacent slices.
- A small U-Net can still learn useful frequency-sensitive features; extra hand-crafted frequency inputs are not automatically advantageous.

**Exact channel-by-channel architecture:** Inspect `src/models/unet.py` as the source of truth. This handbook intentionally does not invent layer widths, kernel sizes, or skip-connection placements that were not captured in the verified discussion. Add an autogenerated model summary after inspecting the file.

---

## 7. Model B — frequency-guided residual U-Net

**Files:** `src/models/frequency_features.py`, `src/models/frequency_unet.py`  
**Constructor:** `FrequencyGuidedUNet(base=16, wavelet_levels=2)`  
**Approximate parameter count reported:** **491,082**.

### 7.1 Sobel features: directional local gradients

Sobel operators estimate horizontal and vertical intensity changes. Conceptually:

\[
G_x = K_x*x,\qquad G_y=K_y*x,
\]

where \(*\) denotes convolution. The two resulting channels indicate local edge strength and orientation.

**Intuition:** Motion ghosting may create repeated or directionally structured edges. **Counterargument:** Meniscus and cartilage boundaries also create strong gradients; Sobel cannot label an edge as artifact versus anatomy.

### 7.2 Haar wavelet features: scale and orientation

A two-level Haar transform decomposes local structure into approximate low-pass and directional detail components. At each level, the commonly named detail bands are **LH, HL, HH**. The implemented two-level feature extraction was reported as **six channels** (three detail bands per level). Combined with two Sobel channels and the input-related channel, the gating module was described as taking **nine channels**; verify the exact concatenation in code.

**Intuition:** Artifacts and anatomy may differ across spatial scales or orientations. **Counterargument:** A wavelet response measures frequency/edge structure, not the causal source of that structure. High-frequency enhancement can emphasize artifacts.

### 7.3 Learned gating and fusion

A learned gate modulates how frequency-derived features influence the U-Net. The idea is to learn when and where these cues are useful instead of applying a fixed sharpening or suppression rule.

**Important:** The current implementation was described as a relatively simple feature gate. It has **not** been shown to identify motion severity, suppress artifact-specific frequencies, or preserve anatomy by design. Those would require additional mechanisms and experiments.

### 7.4 Frequency guidance versus MRI k-space

MRI raw data are acquired in Fourier-encoded k-space, and patient motion can produce phase inconsistencies and structured ghosts. But our model operates on **reconstructed magnitude images** using Sobel/Haar-derived features. Therefore:

- It is reasonable to call this **image-domain frequency-feature guidance**.
- It is **not** a physics-based raw k-space motion correction method.
- It should not be presented as explicitly undoing measured acquisition phase errors.

### 7.5 Fair comparison design

The two models are similar in parameter count (~487k versus ~491k), which helps but does not alone guarantee fairness. Match data split, normalization, seed, epochs, optimizer, learning rate, batch size, evaluation set, preprocessing, and model-selection criterion. Run multiple seeds when feasible.

---

## 8. Loss function, optimizer, and metrics

### 8.1 Training objective

Implemented in `scripts/train_unet.py`:

\[
\mathcal L = \operatorname{L1}(\hat y,y) + 0.2\,[1-\operatorname{SSIM}(\hat y,y)].
\]

**L1** encourages pixel-level agreement and is less sensitive to extreme errors than squared error. **SSIM** encourages similarity in local luminance, contrast, and structure. Their weighted combination balances intensity fidelity and structural appearance, but neither guarantees diagnostic fidelity.

The SSIM implementation was described as differentiable and using a **7×7 Gaussian window**. Verify implementation details (constants, padding, averaging, dynamic range) before comparing values with external libraries.

### 8.2 Optimizer and reported settings

Reported training used **Adam**, learning rate **1e-4**, batch size **2**, `base=16`, CUDA device. An earlier design discussion mentioned AdamW, but **the implemented training code was described as Adam**; the code is authoritative.

### 8.3 PSNR

For normalized images with dynamic range \(MAX=1\):

\[
\operatorname{MSE}=\frac1N\sum_i(\hat y_i-y_i)^2,\qquad
\operatorname{PSNR}=10\log_{10}\frac{MAX^2}{\operatorname{MSE}}.
\]

Higher PSNR means lower pixelwise squared error; it does **not** necessarily mean better clinical anatomy. PSNR values depend on normalization and whether averaging is done per slice or per volume.

### 8.4 SSIM

SSIM compares local mean, variance, and covariance:

\[
\operatorname{SSIM}(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
\]

Higher is generally better, but SSIM can miss clinically important local errors. Report input-to-reference metrics, restored-to-reference metrics, and per-case deltas.

### 8.5 Metric aggregation caveat

The training loop was described as averaging **batch-level** metrics without weighting the last, potentially smaller batch. The standalone evaluation writes per-volume aggregates. Do not compare a training-loop mean and a per-volume mean as if they used identical weighting. For formal reporting, specify aggregation exactly and prefer paired per-volume or true-patient analyses.

---

## 9. Code organization and execution

### 9.1 Source tree at GitHub push

```text
kmar50k_code/
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── samplers.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── kmar_dataset.py
│   │   └── normalization.py
│   └── models/
│       ├── __init__.py
│       ├── unet.py
│       ├── frequency_features.py
│       └── frequency_unet.py
├── scripts/
│   ├── audit_active_pairs.py
│   ├── audit_geometry.py
│   ├── audit_image_pairs.py
│   ├── compare_normalization.py
│   ├── evaluate.py
│   └── train_unet.py
├── manifests/
│   └── dataset_manifest.csv   # LOCAL ONLY, NOT IN GIT
├── checkpoints/             # LOCAL ONLY, NOT IN GIT
└── outputs/                 # LOCAL ONLY, NOT IN GIT
```

The GitHub commit staged **17 files**: `.gitignore`, six scripts, and ten `src` entries (including empty `__init__.py` files). Generated data and checkpoints were excluded.

### 9.2 Training CLI

Representative full-epoch baseline invocation, **run from a compute node with data accessible**:

```bash
python scripts/train_unet.py \
  --model unet \
  --normalization independent \
  --seed 42 \
  --epochs 1 \
  --batch-size 2 \
  --base 16 \
  --lr 1e-4 \
  --max-train-steps 0 \
  --max-val-steps 0 \
  --device cuda \
  --output checkpoints/unet_independent_full_epoch1.pt
```

The command-line `--max-train-steps 0 --max-val-steps 0` means **unlimited steps within that epoch**, not zero training. Earlier short experiments used `--max-train-steps 300` and `--max-val-steps 100`. The script saves checkpoint data including model state, optimizer state, validation loss, base width, epoch, and normalization. A seed/reproducibility patch was applied and `py_compile` succeeded; inspect the current source to confirm exact deterministic flags and seed coverage.

**Important distinction:** Three epochs of 300 training batches are **900 batches total**, not three passes through 20,773 slices. With batch size 2, only approximately 1,800 slice presentations occurred, compared with ~10,387 batches in **one true full epoch**.

### 9.3 Evaluation CLI and output

`scripts/evaluate.py` loads checkpoint metadata, chooses the saved normalization policy, and evaluates slices/volumes on a specified split. `--max-slices 0` was used for full evaluation. Its per-volume CSV includes:

```text
volume_id,slices,psnr,ssim,input_psnr,input_ssim,delta_psnr,delta_ssim
```

Check the script's current argument names with `python scripts/evaluate.py --help` before copying a command from older notes. A formal test run should be performed only after the experiment design is frozen.

---

## 10. Experiment log and exact results

### 10.1 Earlier artifact-only normalization: limited-step training

**Budget:** 3 reported epochs × 300 train batches, batch size 2; validation was limited during training, then a full 227-volume validation evaluation was performed.

| Model | Input PSNR | Restored PSNR | Δ PSNR | Input SSIM | Restored SSIM | Δ SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Residual U-Net | 22.5034 | **22.8993** | +0.3959 | 0.6775 | 0.7157 | +0.0382 |
| Frequency U-Net | 22.5034 | 22.8817 | +0.3784 | 0.6775 | **0.7162** | +0.0387 |

Paired difference, frequency minus baseline, across **227 validation volumes**:

- Mean PSNR: **−0.01757 dB**; bootstrap 95% interval **[−0.049376, +0.015283]**.
- Mean SSIM: **+0.000523**; bootstrap 95% interval **[−0.000385, +0.001432]**.
- Frequency model won on PSNR for **93/227** volumes and on SSIM for **107/227**.

**Interpretation:** No reliable superiority was demonstrated for either architecture under this early protocol; intervals include zero.

### 10.2 Independent normalization: limited-step training

**Budget:** 3 reported epochs × 300 train batches, batch size 2, `lr=1e-4`, `max-val-steps=100` during training; full 227-volume validation afterward.

**Training logs (U-Net):**

| Epoch | Train loss | Val loss | Val PSNR | Val SSIM |
|---|---:|---:|---:|---:|
| 1 | 0.0998 | 0.1191 | 20.42 | 0.6660 |
| 2 | 0.0930 | 0.1179 | 20.44 | 0.6713 |
| 3 | 0.0882 | 0.1174 | 20.42 | 0.6744 |

**Training logs (frequency U-Net):**

| Epoch | Train loss | Val loss | Val PSNR | Val SSIM |
|---|---:|---:|---:|---:|
| 1 | 0.1016 | 0.1202 | 20.40 | 0.6616 |
| 2 | 0.0944 | 0.1194 | 20.30 | 0.6672 |
| 3 | 0.0894 | 0.1186 | 20.37 | 0.6699 |

**Full validation evaluation:**

| Model | Input PSNR | Restored PSNR | Δ PSNR | Input SSIM | Restored SSIM | Δ SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Residual U-Net | 22.5603 | **22.9912** | **+0.4309** | 0.6818 | **0.7242** | **+0.0424** |
| Frequency U-Net | 22.5603 | 22.9188 | +0.3585 | 0.6818 | 0.7213 | +0.0395 |

**Paired differences, frequency minus baseline (227 volumes):**

| Statistic | PSNR | SSIM |
|---|---:|---:|
| Mean difference | **−0.072394 dB** | **−0.002925** |
| Median difference | −0.077027 dB | −0.002627 |
| Frequency model wins | 46/227 (20.3%) | 21/227 (9.3%) |
| Bootstrap 95% CI | [−0.088619, −0.055488] | [−0.003263, −0.002602] |
| Wilcoxon p-value | ~1.054×10⁻¹⁷ | ~2.515×10⁻³³ |

**Interpretation:** In this limited-step, single-run comparison, the baseline outperformed the current frequency-guided model on both metrics. The paired differences are numerically small but consistent across volumes. Statistical significance here is **conditional on the evaluation units and experiment design**; filename-derived groups and possible multiple volumes per patient can violate independence assumptions. This is not a general proof that frequency guidance cannot help.

**Associated output files:** `outputs/unet_independent_full_val.csv`, `outputs/frequency_independent_full_val.csv`, `outputs/independent_model_comparison.csv` (original compute-node scratch; backup availability must be verified).

### 10.3 Independent normalization: one *full* U-Net epoch

**Observed terminal output, 9 November 2026 checkpoint:**

- Checkpoint: `checkpoints/unet_independent_full_epoch1.pt`, **5.7 MB**, modification time **Nov 9 03:35** (as shown in terminal).
- CSV: `checkpoints/unet_independent_full_epoch1.csv`.
- Parameters: `--model unet --normalization independent --seed 42 --epochs 1 --batch-size 2 --base 16 --lr 1e-4 --max-train-steps 0 --max-val-steps 0 --device cuda`.

| Epoch | Train loss | Train PSNR | Train SSIM | Val loss | Val PSNR | Val SSIM |
|---|---:|---:|---:|---:|---:|---:|
| 1 | **0.0935558697** | **22.8848201045** | **0.7355986083** | **0.0938622204** | **23.0686822864** | **0.7333814745** |

**Interpretation:** This is evidence the complete epoch and validation logging reached completion and the checkpoint was written. The later Slurm allocation reached a six-hour wall-time limit, but that does **not** invalidate the already saved metrics. It also does **not** tell us the exact duration of this training run or guarantee all other logs were preserved.

**Do not compare this full-epoch result to the short-budget frequency model as a fair architecture comparison.** The frequency-guided model needs a matched full-epoch run using the same split, seed, normalization, and training budget.

### 10.4 Smoke tests

A 5-batch, one-epoch independent-normalization U-Net smoke run recorded training loss ~0.1479, validation loss ~0.0721, validation PSNR ~28.68 and SSIM ~0.7620 on the small subset. Evaluation of 20 slices from one volume yielded PSNR ~27.4914 versus input ~27.4285, and SSIM ~0.7442 versus input ~0.7425. **These are engineering checks only**, not representative dataset-wide performance.

---

## 11. Statistical interpretation and scientific caution

1. **Paired design:** Evaluate both models on exactly the same volumes and compare per-volume metrics. This removes much of the variation from easy/hard cases.
2. **Confidence intervals:** Bootstrap intervals summarize uncertainty across the sampled evaluation units, but they may be overconfident if multiple volumes correspond to the same patient.
3. **Wilcoxon tests:** Very small p-values do not imply clinically important improvements, and cannot fix data leakage, pseudoreplication, or model-selection bias.
4. **Effect size matters:** −0.072 dB is a small numerical difference even if consistently negative. Include distributions and representative images.
5. **No test-set tuning:** Do not select the best design by repeatedly checking test results. Keep validation for development and test for the final locked comparison.
6. **Metric conventions:** Specify normalization, dynamic range, slice-versus-volume aggregation, interpolation, SSIM implementation, and exclusion criteria.
7. **Reproducibility:** Fix and record seeds, code commit, hardware, package versions, Slurm job ID, dataset manifest hash, model configuration, and checkpoint selection.

---

## 12. Research questions and ablation roadmap

The current result motivates *testing* why fixed frequency features underperform, not asserting a cause.

### 12.1 Hypotheses to test

| Hypothesis | Evidence so far | Discriminating experiment |
|---|---|---|
| Sobel features amplify ghosting/noise | Plausible, unproven | U-Net + Sobel only versus U-Net |
| Haar features help at some scales but hurt others | Plausible, unproven | One-level vs two-level Haar; per-band ablation |
| Gate fusion is too simplistic | Plausible, unproven | Inspect gate maps and compare fusion mechanisms |
| Extra frequency features are redundant with learned convolutions | Plausible, unproven | Match model capacity; ablate inputs/fusion |
| Frequency-domain supervision helps more than feature concatenation | Untested | U-Net + FFT/wavelet loss under matched budget |
| Benefit depends on artifact severity or plane | Untested | Stratified paired evaluation |
| Independent normalization hides intensity error | Possible | Compare deployment-safe normalization policies and unnormalized diagnostics |

### 12.2 Minimum controlled experiment matrix

1. Residual U-Net baseline.
2. U-Net + Sobel only.
3. U-Net + Haar only.
4. U-Net + Sobel + Haar without learned gate.
5. Current gated frequency U-Net.
6. Baseline + frequency-domain auxiliary loss (carefully weighted).
7. Optional adaptive motion-aware fusion after simpler ablations.

Keep training budgets identical, use at least a few seeds if feasible, report runtime and parameter count, and inspect actual restored images. Prefer small, falsifiable changes to large architecture rewrites.

### 12.3 Additional evaluation

- Per-plane results (sagittal/coronal/transverse where metadata supports it).
- Motion-severity bins **only if trustworthy annotations are present and matched**.
- Worst-case degradation and fraction of cases made worse.
- Edge preservation, gradient error, frequency-band error.
- Visual difference maps and artifact residuals.
- Neighboring-slice consistency for 2D outputs.
- Diagnostic structure assessment by qualified reviewers if possible; PSNR/SSIM alone are insufficient.
- Compute cost, inference latency, memory, and reproducibility.

---

## 13. Infrastructure, Slurm, and VS Code

### 13.1 Where the work lives

| Location | Role |
|---|---|
| `ada-gw1` | Login/gateway; VS Code Remote SSH endpoint; **not** the GPU training host |
| `gnode021` | GPU compute node where the original project and dataset were found |
| `/scratch/aashish.a/paired_knee_mri_motion` | Original **node-local** project and dataset; do not assume visible from other nodes |
| `/home2/aashish.a/kmar50k_code` | Recovered source backup for VS Code and Git |
| `/home2/aashish.a/paired_knee_mri_motion` | Accidentally copied home-directory content; **not** the recovered source project |
| `kmar50k` Conda environment | Model training environment used on compute node |
| `knee-smai` / `base` | Other observed environments; do not assume they match KMAR dependencies |

Reported GPU: **NVIDIA GTX 1080 Ti, 11 GB**, with CUDA-capable PyTorch `2.3.1+cu118` in the training environment. Package versions and hardware should be captured again when producing reproducibility artifacts.

### 13.2 Slurm history and job termination

On 10 November, `squeue` showed no active jobs. `sacct` showed job **65482** as `TIMEOUT` after **06:00:10**, and its step was cancelled. This indicates the **allocation** ended at the six-hour limit, not necessarily that the earlier saved checkpoint was invalid.

A new interactive allocation **67708** successfully targeted `gnode021` with `--partition=u22 --nodelist=gnode021 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash`. This allowed recovery of source files and the 5.7 MB full-epoch checkpoint. Such allocations are temporary and can expire.

**Operational rule:** Long training should run as a Slurm batch job with frequent checkpoints, resumable optimizer/sampler/RNG state, progress logging, and outputs written to durable storage. Do not rely on an interactive terminal remaining open.

### 13.3 VS Code Remote SSH and storage incident

VS Code initially failed with `Failed to parse remote port from server output`. Diagnostics showed `/home2/aashish.a` at **100% capacity and 100% inode usage**; `~/.vscode-server` used about **2.5 GB**, `~/.cache` about **7.1 GB**, and a mistaken 13 GB copy occupied `/home2/aashish.a/paired_knee_mri_motion`.

Running `python -m pip cache purge` removed **758 files (~4,926.4 MB)** and **1,413 directories**. Follow-up diagnostics showed **4.7 GB free**, **91% used**, and ~**9.66 million free inodes**. VS Code Remote SSH then connected successfully. When VS Code asks for the remote OS, select **Linux**, despite Windows being the local workstation.

**Safety lesson:** Never copy an entire home directory into the project by mistake. Verify `pwd`, source, destination, and excludes before `rsync`. Avoid deleting `.vscode-server`, Conda, datasets, or checkpoints indiscriminately.

---

## 14. Git and repository history

**Target repository:** https://github.com/anandaashish74711/-KMAR-50K-Motion-Restoration
**Default branch observed:** `origin/main`  
**Feature branch:** `feature/kmar50k-motion-restoration`.

Sequence:

1. `/home2/aashish.a/kmar50k_code` was not initially a Git repository.
2. Initialized Git and configured `origin` to the target HTTPS repository.
3. `git fetch origin` succeeded and discovered `origin/main`.
4. Initial `git switch -c ... origin/main` was blocked by an existing untracked `.gitignore`; the local file was backed up and its ignore rules appended after switching.
5. Created `feature/kmar50k-motion-restoration` from `origin/main`.
6. Staged the source scripts, model/data modules, and `.gitignore`.
7. Unstaged `manifests/dataset_manifest.csv` because it contained cluster-specific absolute paths and potentially sensitive identifiers; added it to `.gitignore`.
8. Excluded raw MRI volumes, checkpoints, temporary files, and generated outputs.
9. `git push -u origin feature/kmar50k-motion-restoration` **succeeded**; remote branch created and upstream tracking set. The remote `main` branch was **not changed**.


---

## 15. What went wrong and what we learned

### 15.1 Original normalization clipped clean targets

Using the artifact's intensity percentiles for both images sometimes saturated large portions of the reference. This could teach the model a distorted target. We implemented independent scaling and recorded the normalization in checkpoints. But independent scaling also changes the scientific interpretation of intensity fidelity.

### 15.2 “Three epochs” did not mean three full dataset passes

With a 300-step cap, the reported epochs were deliberately truncated. Early architecture comparisons therefore tested **early optimization under a small budget**, not final converged performance. The later baseline experiment explicitly used zero caps to process all batches.

### 15.3 Frequency features did not outperform the baseline

The initial hypothesis was not supported for the current gated Sobel/Haar architecture. Rather than adding complexity blindly, we should isolate features and fusion mechanisms, examine failure cases, and test targeted improvements.

### 15.4 Node-local scratch nearly became a single point of failure

The source lived on `gnode021` and disappeared from the gateway's view when the Slurm allocation expired. `pam_slurm_adopt` correctly denied SSH without an active job. A new authorized Slurm allocation recovered the source. The code was then backed up to shared storage and GitHub.

### 15.5 Backup copied the wrong directory

An earlier `rsync` inadvertently copied home-directory contents into a misleadingly named project folder, consuming quota and leaving no `src/models` there. We created the clean `kmar50k_code` directory and verified `src/models` before proceeding.

### 15.6 VS Code failed because of quota, not the Windows OS selection

Remote host is Linux. Storage exhaustion prevented remote server startup; purging pip cache restored enough capacity.

### 15.7 Git protected the untracked `.gitignore`

Switching to `origin/main` initially failed because checkout would overwrite the local ignore file. Git's protection was correct. We preserved the ignore rules rather than forcing checkout or deleting project files.

---

## 16. Outstanding risks, ranked

| Priority | Risk | Why it matters | Next action |
|---|---|---|---|
| Critical | Filename groups are not verified patient IDs | Potential train/val/test leakage | Obtain authoritative patient mapping or verify dataset grouping |
| Critical | Absolute-path manifest excluded from Git | Fresh clone cannot run | Configurable data root + manifest builder/rewriter |
| High | Mid-epoch resume not implemented | Six-hour allocations can waste training | Save model/optimizer/RNG/sampler/progress regularly |
| High | Frequency full-epoch control missing | No fair full-budget model comparison | Train matched frequency run, ideally multiple seeds |
| High | Clinical structure preservation not validated | PSNR/SSIM can hide anatomy errors | Targeted qualitative/clinical evaluation |
| High | Geometry-review cases excluded | Selection bias / reduced data | Document exclusions; investigate safe registration |
| Medium | Per-slice square resizing | Physical anisotropy/shape distortion | Record interpolation and test geometry-aware preprocessing |
| Medium | Intensity normalization policy | Cross-policy metric comparisons misleading | Lock policy per experiment; compare alternatives |
| Medium | Per-volume statistical independence | Potentially overconfident confidence intervals | Patient-level clustering when IDs available |
| Medium | No clean environment lock | Harder to reproduce | `requirements.txt`/environment YAML + versions |
| Medium | Full training cost and convergence unknown | Budget planning and model selection | Benchmark per-epoch time, learning curves |

---

## 17. Suggested next development sequence

**Phase A — Make the current project reproducible**

1. Inspect the pushed source in VS Code; generate a precise model summary and verify all CLI options.
2. Implement configurable `--data-root` and a manifest path resolver/generator without exposing absolute paths.
3. Add a minimal environment specification and a README with expected dataset layout.
4. Add periodic checkpointing and resume support, including RNG state and batch progress where practical.
5. Add structured experiment IDs, metrics JSON/CSV, Git commit hash, Slurm job ID, and config snapshots.
6. Verify `VolumeShuffleSampler` covers all indices once and test loader outputs/normalization on synthetic arrays.

**Phase B — Understand and challenge every component**

1. Trace one pair from NIfTI header to 2D tensor to U-Net output.
2. Draw the actual U-Net layer graph from `unet.py` and calculate tensor shapes and parameter counts.
3. Derive Sobel and Haar features on small numeric examples; visualize them on real MRI slices.
4. Inspect gate activations and identify whether they respond to anatomy or artifacts.
5. Reproduce L1, SSIM, PSNR, and per-volume aggregation calculations manually for one slice.

**Phase C — Rigorous experiments**

1. Run matched full-epoch baseline and frequency-guided training, using the same seed and configuration.
2. Extend training with consistent stopping/checkpoint selection and several seeds.
3. Run feature and loss ablations; stratify results by plane/severity where metadata permits.
4. Inspect worst regressions, improvements, difference maps, and anatomy preservation.
5. Freeze the model selection protocol, then evaluate the held-out test set once.
6. Write the final scientific report with both positive and negative findings.

---

## 18. Reproduction checklist

- [ ] Confirm dataset release, licensing, metadata, and exact pair counts.
- [ ] Confirm verified patient-level split or explicitly state filename-group limitation.
- [ ] Recreate or relocate manifest using configurable data root.
- [ ] Confirm all active pairs have acceptable geometry and no overlap.
- [ ] Reproduce P1/P99 normalization tests and extreme clipping cases.
- [ ] Verify model parameter counts and actual layer architecture.
- [ ] Record package versions, CUDA, GPU, seed, Git SHA, and Slurm job ID.
- [ ] Verify sampler index coverage.
- [ ] Re-run one full baseline epoch and match saved metrics.
- [ ] Run a matched full frequency epoch.
- [ ] Reproduce paired validation comparisons with clearly defined units.
- [ ] Evaluate clinical/structural preservation and failures.
- [ ] Keep test split untouched until decisions are frozen.
- [ ] Save configs, logs, plots, and checkpoint hashes outside ephemeral scratch.
- [ ] Document all exclusions and limitations in final report.

---

## 19. Command reference

### Open code in VS Code Remote SSH

```bash
# In VS Code: Remote-SSH to ada-gw1 (Linux)
cd /home2/aashish.a/kmar50k_code
pwd
ls src/models scripts
```

### Inspect Git branch

```bash
git branch --show-current
git status
git remote -v
```

### Commit documentation and future code changes

```bash
git add docs/ README.md  # only if these files exist and are intended for commit
git diff --cached --stat
git commit -m "Document KMAR-50K pipeline and experiments"
git push
```

### Inspect Slurm jobs

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime 2026-11-09 --format=JobID,JobName,State,Elapsed,ExitCode
```

### Check quota

```bash
quota -s
df -h "$HOME"
df -i "$HOME"
```

### Verify checkpoint from a compute node with the original scratch

```bash
ls -lh /scratch/aashish.a/paired_knee_mri_motion/checkpoints/unet_independent_full_epoch1.pt
cat /scratch/aashish.a/paired_knee_mri_motion/checkpoints/unet_independent_full_epoch1.csv
```

**Do not run GPU training on `ada-gw1`.** Request a permitted Slurm allocation or submit a batch job. Data may be node-local to `gnode021`.

---

## 20. Glossary

**Artifact:** Unwanted image distortion, here arising from motion during MRI acquisition.  
**Affine:** Mapping from image voxel indices to physical coordinates.  
**NIfTI:** Neuroimaging file format storing image volumes and spatial metadata.  
**Pair:** Corresponding motion-affected and reference image volumes.  
**P1/P99:** First and ninety-ninth intensity percentiles.  
**Residual learning:** Predicting a correction to the input rather than the entire output.  
**U-Net:** Encoder–decoder CNN with skip connections.  
**Sobel:** Spatial gradient filter estimating directional intensity changes.  
**Haar wavelet:** Multiscale filter bank separating approximate and directional detail signals.  
**Gate:** Learned modulation of feature contributions.  
**k-space:** Fourier-encoded MRI acquisition domain; not directly used by the implemented models.  
**L1 loss:** Mean absolute pixel error.  
**SSIM:** Structural similarity metric.  
**PSNR:** Peak signal-to-noise ratio derived from MSE.  
**Ablation:** Controlled removal/change of one component to estimate its contribution.  
**Checkpoint:** Saved model/training state.  
**Slurm:** HPC job scheduler controlling compute-node allocations.  
**Remote SSH:** VS Code editing over an SSH connection to the cluster gateway.  
**Group leakage:** Related scans crossing dataset splits and inflating estimated generalization.

---

## 21. Provenance and verification notes

This handbook synthesizes the experiment logs, command outputs, dataset audits, implementation descriptions, and decisions discussed during project development. It is **not** a line-by-line independent audit of the currently pushed source files or the full dataset publication. Specific implementation details, especially exact model blocks, image axes, interpolation, optimizer behavior, SSIM constants, seed handling, and manifest generation, should be checked against the repository and annotated in a later revision. Historical numerical results are transcribed as previously reported; preserve their CSV logs and code versions for verification.

**Last recorded milestone:** GitHub push of `feature/kmar50k-motion-restoration` succeeded on 10 November 2026. The repository `main` branch remained unchanged. No later architecture changes, documentation commits, or experiments are claimed here.
