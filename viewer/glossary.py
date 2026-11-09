"""Plain-language glossary for the KMAR-50K viewer."""

GLOSSARY = {
    "The project in one paragraph": [
        ("What are we trying to do?",
         "Patients sometimes move during an MRI scan, which leaves **motion artifacts** (ghosting, blur) in "
         "the image. KMAR-50K gives us pairs of *the same knee*: one scan ruined by motion (**X**) and an "
         "immediate rescan without motion (**Y**). We train a model that takes X and outputs an estimate "
         "**Ŷ** close to Y. Our twist: besides X itself, the model also receives hand-crafted image-processing "
         "features — gradients **G(X)**, multi-scale bands **H(X)** and directional responses **D(X)** — and "
         "we test whether these help compared with a model that only sees X."),
        ("X, Y, R, Ŷ, ΔX",
         "* **X** — motion-corrupted image (model input).\n"
         "* **Y** — clean reference (rescan).\n"
         "* **R = X − Y** — residual; ideally the artifact itself.\n"
         "* **ΔX** — the correction the network predicts.\n"
         "* **Ŷ = X + ΔX** — the restored image (*residual learning*)."),
        ("Residual learning",
         "Instead of asking the network to redraw the whole knee, we ask it to predict only the *correction* "
         "ΔX and add it to the input. Most of the image is already correct, so this is an easier task and "
         "tends to preserve anatomy. Our baseline **M2** uses residual learning too, so any gain of the "
         "proposed model **M3** cannot be explained by residual learning alone."),
    ],
    "MRI acquisition": [
        ("MRI (Magnetic Resonance Imaging)",
         "Imaging that uses a strong magnetic field and radio waves to measure signals from hydrogen nuclei "
         "(mostly in water and fat). Different tissues give different signal strengths, which become the "
         "grey levels of the image. No ionising radiation."),
        ("k-space",
         "MRI scanners do not take a photo directly — they measure the image's **Fourier transform** (its "
         "frequency content), called k-space, and the image is reconstructed by an inverse Fourier transform. "
         "The centre of k-space holds contrast and coarse shape; the outer parts hold fine detail and edges."),
        ("Frequency-encode vs phase-encode direction",
         "k-space is filled **line by line**. Along a line (the *frequency-encode* direction) data are read "
         "in milliseconds. Successive lines (the *phase-encode* direction) are acquired seconds to minutes "
         "apart. Motion therefore makes **different lines inconsistent** with each other, which corrupts the "
         "image mostly **along the phase-encode direction**."),
        ("Motion artifact / ghosting",
         "When the knee moves between k-space lines, the reconstruction produces **ghosts**: faint shifted "
         "replicas of the anatomy repeating along the phase-encode direction. Other effects: **blur** (edges "
         "smeared), **ringing** and **signal loss**. Unlike random noise, these are *structured* and can look "
         "like real anatomy, which is why they are hard to remove."),
        ("Retrospective motion correction",
         "Fixing the artifact *after* the scan from the image data, as opposed to *prospective* correction "
         "(tracking the patient during the scan) or simply rescanning. This project is retrospective and "
         "image-domain (we work on reconstructed images, not raw k-space)."),
        ("Field strength (1.5 T / 3 T)",
         "Strength of the scanner's main magnet in tesla. 3 T gives more signal (sharper / less noisy images) "
         "than 1.5 T. Most KMAR-50K scans are 1.5 T (Siemens Aera)."),
    ],
    "Sequences and protocol names": [
        ("Pulse sequence / weighting",
         "The recipe of radio pulses and timings the scanner uses. It controls which tissue appears bright. "
         "Changing the sequence changes the contrast, so models are usually trained on one sequence at a time."),
        ("PD (proton density) weighting",
         "Contrast mainly reflects how many hydrogen nuclei are present. Standard for knee imaging because it "
         "shows cartilage, menisci and ligaments well."),
        ("T1 weighting",
         "Contrast reflects T1 relaxation time. Fat is bright, fluid is dark; good for anatomy and bone marrow."),
        ("FS (fat suppression / fat saturation)",
         "An extra pulse that cancels the fat signal so fat appears dark. With PD, this makes fluid and "
         "oedema stand out. 'pd_tse_fs' = PD-weighted, turbo spin echo, fat-suppressed."),
        ("TSE (Turbo Spin Echo)",
         "A fast spin-echo sequence that acquires several k-space lines per excitation. Shorter than classic "
         "spin echo, but the scan still lasts minutes, so motion can occur."),
        ("TR (Repetition Time) / TE (Echo Time)",
         "**TR**: time between successive excitation pulses (ms). **TE**: time from excitation to signal "
         "readout (ms). Together they set the contrast. PD weighting uses long TR and short TE."),
        ("Flip angle",
         "How far the radio pulse tips the magnetisation (degrees). TSE sequences typically use ~120–180° "
         "refocusing pulses — the values in the metadata."),
        ("Protocol suffixes (_tra/_sag/_cor, _256/_320, _p2, _r/_l)",
         "Siemens protocol naming: **tra/sag/cor** = plane; **256/320** = acquisition matrix size; **p2** = "
         "parallel imaging with acceleration factor 2 (faster scan); **r/l** most likely right/left knee. "
         "A leading **q** appears in some names; its meaning is site-specific (treated as the same family here)."),
    ],
    "Image geometry and file format": [
        ("Planes: sagittal, coronal, transverse (axial)",
         "* **Sagittal** — slices from side to side (you see the knee from the side).\n"
         "* **Coronal** — slices from front to back (you see the knee from the front).\n"
         "* **Transverse / axial** — horizontal slices (looking down the leg). KMAR-50K calls this "
         "**transection**."),
        ("Slice, volume, slice thickness",
         "A **volume** is a stack of 2-D **slices**. KMAR-50K volumes have ~20–25 slices, each **3–5 mm thick**, "
         "while pixels inside a slice are much smaller (~0.25–0.7 mm). Because slices are thick and far apart, "
         "we restore each slice in 2-D."),
        ("Pixel spacing / voxel size",
         "Physical size of one pixel (in-plane, mm) and one voxel (3-D pixel: in-plane × slice thickness)."),
        ("NIfTI (.nii.gz)",
         "Standard neuro/medical image file format: a 3-D array of intensities plus a header with voxel size "
         "and an **affine** matrix mapping array indices to positions in the scanner (mm)."),
        ("Orientation codes (e.g. PIR, LPS)",
         "Three letters telling which anatomical direction each array axis points towards: L/R = left/right, "
         "A/P = anterior/posterior (front/back), S/I = superior/inferior (head/feet). The viewer uses them to "
         "display slices upright."),
        ("Intensity / grey value",
         "The number stored per pixel. Raw MRI intensities are arbitrary scanner units (here ~0–1,500), not "
         "physical quantities, so every volume must be **normalised** before comparison."),
    ],
    "Preprocessing": [
        ("Normalisation",
         "Rescaling intensities to a common range. We divide each pair by the clean volume's **99.5th "
         "percentile** (robust to a few very bright pixels) so X and Y share the same scale and values are ~0–1."),
        ("N4 bias field correction",
         "MRI images often have a smooth brightness gradient caused by the receive coil (the *bias field*), "
         "not by anatomy. **N4** (an algorithm from the ANTs/ITK toolkits) estimates and removes it. Files "
         "with **_N4** have been corrected."),
        ("Registration (alignment) / SyN",
         "Moving one image so its anatomy lines up with another. Needed because the patient's knee is not in "
         "exactly the same position in the rescan. **SyN** (Symmetric Normalization, ANTs) is a *deformable* "
         "registration — it can bend the image locally, not just shift/rotate it. Files with **_RegSyN** were "
         "registered this way."),
        ("Misregistration",
         "Remaining misalignment between X and Y. It inflates the error of *every* method (the model is "
         "penalised for not reproducing a shift it cannot know about), so we measure it and drop bad pairs."),
        ("Foreground mask",
         "A binary map of where tissue is (vs. black background), obtained by thresholding. Used to drop "
         "near-empty slices and to compute metrics only inside the knee."),
        ("Patient-level split",
         "All scans of one patient go into the same subset (train, val or test). Otherwise a model could "
         "*memorise* a patient's anatomy from training and look artificially good on test (**data leakage**)."),
        ("Data augmentation",
         "Random, label-preserving transformations of training images (e.g. horizontal flip, small rotation) "
         "to reduce overfitting. We avoid vertical flips / 90° rotations because they would change the "
         "phase-encode direction of the ghosts."),
    ],
    "Digital image processing tools": [
        ("Fourier transform (2-D FFT)",
         "Expresses an image as a sum of stripe patterns (sinusoids) of different **frequencies** (how fine) "
         "and **orientations** (which way). The **FFT** is the fast algorithm to compute it. We plot "
         "**log|F|** (log magnitude) with the zero frequency (**DC**, average brightness) in the centre."),
        ("Spatial frequency: low vs high",
         "**Low** frequencies = slowly varying content (overall shape, contrast). **High** frequencies = rapid "
         "changes (edges, fine texture, noise). Key issue for this project: **anatomy edges and motion "
         "artifacts both contain high frequencies**, so removing all high frequencies blurs the anatomy."),
        ("ΔF (spectral difference)",
         "ΔF(u,v) = |F_X(u,v)| − |F_Y(u,v)| (Eq. 5 of the proposal; the viewer shows it on a log scale). "
         "Positive = frequencies the artifact *added*, negative = frequencies motion *removed* (blur)."),
        ("Radial energy profile",
         "Average spectral power as a function of distance from the spectrum centre. Summarises \"how much "
         "content is at each scale\" regardless of direction."),
        ("Angular / directional energy",
         "Spectral power summed over wedge-shaped sectors (directions). Tells us whether content (or the "
         "artifact) is concentrated in particular orientations."),
        ("Convolution / filter / kernel",
         "Sliding a small weight pattern (the **kernel**) over the image and taking a weighted sum at each "
         "position. Blurring, edge detection, Gabor filtering and the layers of a CNN are all convolutions."),
        ("Gradient / Sobel operator — G(X)",
         "The gradient is the rate of brightness change in x and y; its **magnitude** highlights edges and "
         "its **orientation** says which way the edge runs. **Sobel** is a 3×3 kernel that estimates it with "
         "slight smoothing."),
        ("Gaussian pyramid / Laplacian pyramid — H(X)",
         "**Gaussian pyramid**: repeatedly blur and downsample (each level half the size). **Laplacian "
         "pyramid**: the difference between consecutive Gaussian levels — each band holds the detail of one "
         "scale, plus a final low-pass image. The bands sum back to the original exactly."),
        ("Wavelet transform (LL, LH, HL, HH)",
         "Splits the image into a coarse approximation (**LL**) and detail sub-bands per orientation: "
         "**LH** horizontal, **HL** vertical, **HH** diagonal detail — repeated over several scales. Used for "
         "multi-scale analysis and **wavelet denoising** (shrinking small detail coefficients)."),
        ("Gabor filter — D(X)",
         "A sinusoidal stripe pattern inside a Gaussian window, tuned to one **orientation θ** and one "
         "**frequency**. A bank of Gabors at several θ gives a directional decomposition D_θ(X) (Eq. 7)."),
        ("Steerable / directional filters",
         "General term for filters whose orientation can be chosen (Gabor, oriented derivatives). They "
         "respond strongly to structures aligned with their orientation."),
    ],
    "Classical restoration baselines": [
        ("Gaussian filter",
         "Blurs by averaging with Gaussian weights (parameter σ = blur width). Removes noise and ghosts but "
         "also softens edges — the classic trade-off."),
        ("Median filter",
         "Replaces each pixel by the median of its neighbourhood (k×k). Removes isolated outliers while "
         "keeping edges better than Gaussian blur."),
        ("Wiener filter",
         "An adaptive filter that smooths more where local variance is low (flat areas) and less near edges, "
         "based on an estimate of signal vs noise power."),
        ("Frequency-domain filtering (Butterworth, notch, directional)",
         "Multiply the Fourier spectrum by a mask and invert. **Butterworth low-pass** smoothly removes high "
         "frequencies; a **directional/notch** filter suppresses specific orientations or frequencies — e.g. "
         "the phase-encode band where ghosts live."),
        ("Wavelet denoising (BayesShrink / VisuShrink)",
         "Transform to wavelets, shrink small detail coefficients towards zero (they are mostly noise), "
         "transform back. BayesShrink and VisuShrink are two rules for choosing the threshold."),
        ("Non-local means",
         "Denoises each pixel by averaging pixels elsewhere in the image whose *surrounding patches* look "
         "similar. Preserves repeated structure well."),
    ],
    "Learned models": [
        ("U-Net",
         "A convolutional neural network shaped like a U: an **encoder** repeatedly downsamples to capture "
         "context, a **decoder** upsamples back to full resolution, and **skip connections** pass fine detail "
         "across. The standard architecture for image-to-image tasks in medical imaging (Ronneberger 2015)."),
        ("M1 / M2 / M3",
         "* **M1** — image-only U-Net, predicts Ŷ directly.\n"
         "* **M2** — image-only *residual* U-Net, predicts ΔX.\n"
         "* **M3 (proposed)** — same residual U-Net, input Z = [X, G(X), H(X), D(X)] (16 channels)."),
        ("Ablation",
         "Removing or changing one component at a time to measure its contribution — e.g. X+G only, X+H only, "
         "X+D only — and control runs (random filters, learnable stem) to rule out trivial explanations."),
        ("Loss function (L1, SSIM loss)",
         "What training minimises. **L1** = mean absolute error; **1 − SSIM** penalises structural "
         "differences. We use L1 + 0.2·(1 − SSIM)."),
        ("Train / validation / test",
         "**Train**: fit model weights. **Validation**: choose hyper-parameters and the best checkpoint. "
         "**Test**: report final numbers once, never used for decisions."),
    ],
    "Evaluation metrics": [
        ("PSNR (Peak Signal-to-Noise Ratio)",
         "10·log10(MAX² / MSE), in dB. Higher = closer to the reference. Every +3 dB halves the squared error. "
         "Sensitive to brightness offsets; insensitive to *where* errors are."),
        ("SSIM (Structural Similarity Index)",
         "Compares local luminance, contrast and structure in small windows; 1 = identical. Better matches "
         "perceived quality than PSNR (Wang et al. 2004)."),
        ("MAE / MSE / NRMSE",
         "**MAE** mean |error|. **MSE** mean squared error. **NRMSE** = √MSE divided by the reference's RMS "
         "intensity — scale-free, lower is better."),
        ("NCC (Normalised Cross-Correlation)",
         "Pearson correlation of two images (−1…1). Used to measure X–Y alignment in the pair-quality check."),
        ("Mutual information",
         "How much knowing one image's intensities tells you about the other's. A registration-quality "
         "measure that tolerates contrast differences."),
        ("HFEN (High-Frequency Error Norm)",
         "Error after filtering both images with a Laplacian-of-Gaussian (an edge detector). Measures how "
         "well **fine edges** are reproduced — standard in MRI reconstruction."),
        ("GMSD / edge F1",
         "**GMSD**: deviation of gradient-magnitude similarity (edge-based quality). **Edge F1**: run an edge "
         "detector (Canny) on Ŷ and Y and score overlap. Both check that restoration does not erase anatomy."),
        ("Radial spectral error / band error",
         "Compare radial energy profiles (or wavelet bands) of Ŷ and Y: tells *at which scale* a method "
         "still differs — e.g. Gaussian filtering will lose high-frequency energy."),
        ("Paired Wilcoxon test / bootstrap CI",
         "Statistical tools to check that a difference between two methods on the same test patients is real "
         "rather than chance. Computed per patient, not per slice, to avoid overstating significance."),
    ],
}
