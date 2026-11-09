"""KMAR-50K dataset viewer — explore paired motion-corrupted / clean knee MRI.

Run:
    /archive/simran.singh/envs/simran_kmar50/bin/streamlit run viewer/app.py --server.port 8790
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pywt  # noqa: E402
import streamlit as st  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

from glossary import GLOSSARY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kmar50k"
META = ROOT / "data" / "raw" / "meta"
NAME_RE = re.compile(r"^(?P<patient>.+?)_(?P<plane>sagittal|coronal|transection)_(?P<series>\d+(?:\.\d+)?)"
                     r"(?P<suffix>_N4_Norm(?:_RegSyN)?)?\.nii(?:-\w+)?\.gz$")
PLANE_LABEL = {"sagittal": "Sagittal", "coronal": "Coronal", "transection": "Transverse (axial)"}

st.set_page_config(page_title="KMAR-50K Viewer", layout="wide")


# ----------------------------------------------------------------------------- data
def seq_family(desc):
    d = str(desc).lower().strip().strip("'").strip()
    m = re.match(r"^q?(pd|t1|t2)_tse(_fs)?_(tra|sag|cor)", d)
    if not m:
        return "other"
    return m.group(1).upper() + ("-FS" if m.group(2) else "")


@st.cache_data
def load_meta():
    rows = []
    for fn, col in [("TrainingCohort.csv", "NII_FileName"), ("TestingCohort.csv", "Patient")]:
        p = META / fn
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["csv_key"] = df[col].astype(str).str.replace(".nii.gz", "", regex=False).str.strip()
        df["cohort"] = "train" if fn.startswith("Train") else "test"
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    clean = lambda s: str(s).strip().strip("'").strip()  # noqa: E731
    for c in ["SeriesDescription", "MagneticFieldStrength", "SequenceName", "RepetitionTime", "EchoTime",
              "FlipAngle", "SliceThickness", "PixelSpacing", "Manufacturer", "ManufacturerModel", "StudyDate"]:
        df[c] = df[c].map(clean)
    df["plane"] = df.csv_key.str.extract(r"_(sagittal|coronal|transection)_")[0]
    df["family"] = df.SeriesDescription.map(seq_family)
    df["field_T"] = df.MagneticFieldStrength.astype(float)
    return df.drop_duplicates("csv_key").set_index("csv_key")


def scan_pairs():
    """Pairs whose artifact AND clean files are both on disk (works during a partial download)."""
    out = []
    for split in ["train", "test"]:
        files = {}
        for kind in ["artifact", "clean"]:
            d = RAW / split / kind
            if not d.exists():
                continue
            for p in d.glob("*.gz"):
                m = NAME_RE.match(p.name)
                if not m:
                    continue
                g = m.groupdict()
                variant = "processed" if g["suffix"] else "raw"
                key = (g["patient"], g["plane"], g["series"], variant)
                files.setdefault(key, {})[kind] = str(p)
        for (pat, plane, series, variant), kinds in files.items():
            if len(kinds) == 2:
                out.append(dict(split=split, patient=pat, plane=plane, series=series, variant=variant,
                                csv_key=f"{pat}_{plane}_{int(float(series))}",
                                art=kinds["artifact"], clean=kinds["clean"]))
    return pd.DataFrame(out)


@st.cache_data(ttl=120)
def pairs_df():
    df = scan_pairs()
    if df.empty:
        return df
    meta = load_meta()
    df["family"] = df.csv_key.map(meta["family"]).fillna("?")
    df["sequence"] = df.csv_key.map(meta["SeriesDescription"]).fillna("?")
    df["field_T"] = df.csv_key.map(meta["field_T"])
    return df.sort_values(["split", "plane", "patient", "series"]).reset_index(drop=True)


@st.cache_data(max_entries=8)
def load_vol(path):
    img = nib.load(path)
    return (np.asarray(img.dataobj, dtype=np.float32),
            tuple(float(z) for z in img.header.get_zooms()[:3]),
            "".join(nib.aff2axcodes(img.affine)), str(img.get_data_dtype()))


def to_display(sl):
    """NIfTI slice (axis0, axis1) -> image rows=axis1, cols=axis0 (anatomically upright for this dataset)."""
    return sl.T


# ----------------------------------------------------------------------------- DIP helpers
def log_spec(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    return F, np.log1p(np.abs(F))


def radial_profile(power, nbins=64):
    h, w = power.shape
    yy, xx = np.indices(power.shape)
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))  # 0 centre .. 1 at Nyquist (edge)
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(r.ravel(), bins) - 1, 0, nbins)
    s = np.bincount(idx, power.ravel(), nbins + 1)[:nbins]
    c = np.bincount(idx, minlength=nbins + 1)[:nbins]
    return (bins[:-1] + bins[1:]) / 2, s / np.maximum(c, 1)


def angular_energy(power, nsec=12, rmin=0.05):
    h, w = power.shape
    yy, xx = np.indices(power.shape)
    dy, dx = (yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2)
    r = np.hypot(dy, dx)
    ang = np.mod(np.arctan2(dy, dx), np.pi)  # spectrum is symmetric -> [0, pi)
    m = (r > rmin) & (r <= 1)
    idx = np.minimum((ang[m] / np.pi * nsec).astype(int), nsec - 1)
    e = np.bincount(idx, power[m], nsec)
    return (np.arange(nsec) + 0.5) * np.pi / nsec, e / e.sum()


def laplacian_pyramid(img, levels=4):
    bands, cur = [], img
    for _ in range(levels - 1):
        low = ndi.gaussian_filter(cur, 1.0)
        down = low[::2, ::2]
        up = ndi.zoom(down, (cur.shape[0] / down.shape[0], cur.shape[1] / down.shape[1]), order=1)
        bands.append(cur - up)
        cur = down
    bands.append(cur)
    return bands


def gabor_bank(img, n_orient=4, freq=0.15, sigma=3.0):
    out = []
    half = int(3 * sigma)
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    for k in range(n_orient):
        th = k * np.pi / n_orient
        xr = x * np.cos(th) + y * np.sin(th)
        g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        re_, im_ = g * np.cos(2 * np.pi * freq * xr), g * np.sin(2 * np.pi * freq * xr)
        re_ -= re_.mean()
        out.append((np.degrees(th), np.hypot(ndi.convolve(img, re_), ndi.convolve(img, im_))))
    return out


def metrics(x, y, mask):
    rng = 1.0
    mse = np.mean((x - y) ** 2)
    psnr = 10 * np.log10(rng ** 2 / max(mse, 1e-12))
    ssim = structural_similarity(x, y, data_range=rng)
    nrmse = np.sqrt(mse) / max(np.sqrt(np.mean(y ** 2)), 1e-12)
    mae = np.mean(np.abs(x - y))
    xm, ym = x[mask], y[mask]
    ncc = float(np.corrcoef(xm, ym)[0, 1]) if mask.sum() > 10 else np.nan
    return dict(PSNR_dB=psnr, SSIM=ssim, NRMSE=nrmse, MAE=mae, NCC_foreground=ncc)


def show(ax, img, title, cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def fig_row(n, h=3.6):
    fig, axs = plt.subplots(1, n, figsize=(h * n, h))
    return fig, np.atleast_1d(axs)


def explain(text):
    with st.expander("What am I looking at?", expanded=False):
        st.markdown(text)


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.header("Dataset overview — KMAR-50K")
    st.markdown("""
**KMAR-50K** (Xi et al., *Scientific Data* 2025) contains **paired knee MRI**: for each case, a scan that
was **corrupted by patient motion** and a **rescan of the same knee** acquired afterwards without motion,
taken in the same session. The rescan acts as the *clean reference* (ground truth) for restoration.

* **Training cohort** — scans from 2020–2022 (1,341 pairs). We will split this by patient into train/val.
* **Testing cohort** — scans from 2023 (103 pairs), released by the authors as a held-out test set.
  It ships in two variants: **raw** and **processed** (`_N4_Norm` / `_N4_Norm_RegSyN` — bias-corrected,
  normalized and registered).
* Each file is a 3-D **NIfTI** volume: a stack of ~20–25 thick 2-D slices of one knee in one plane.
""")
    meta = load_meta()
    pairs = pairs_df()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs in metadata", len(meta))
    c2.metric("Patients (metadata)", meta.index.str.replace(r"_(sagittal|coronal|transection)_\d+$", "",
                                                             regex=True).nunique())
    c3.metric("Pairs downloaded so far", 0 if pairs.empty else len(pairs))
    c4.metric("Test pairs downloaded", 0 if pairs.empty else int((pairs.split == "test").sum()))

    st.subheader("Pairs by sequence family × plane (from the authors' CSVs)")
    tab = pd.crosstab([meta.cohort, meta.family], meta.plane.map(PLANE_LABEL), margins=True)
    st.dataframe(tab, width="stretch")
    explain("""
* **Sequence family** groups the scanner protocol names: **PD-FS** = proton-density weighted with fat
  suppression (the standard knee protocol — bright fluid, dark fat), **T1** = T1-weighted (bright fat, good
  anatomy). See the *Glossary* page.
* **Plane** is the direction the slices are cut: sagittal (side view), coronal (front view), transverse /
  axial (top-down; the dataset calls it *transection*).
* Our project uses **one family + one plane** to keep the comparison controlled. The largest group is
  PD-FS, and the paper's U-Net benchmark is reported on the transverse plane.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Field strength & scanner")
        st.dataframe(pd.crosstab(meta.ManufacturerModel, meta.MagneticFieldStrength), width="stretch")
    with c2:
        st.subheader("Slice thickness (mm)")
        st.dataframe(meta.SliceThickness.value_counts().rename("pairs"), width="stretch")

    st.subheader("Full metadata table")
    st.dataframe(meta[["cohort", "plane", "family", "SeriesDescription", "MagneticFieldStrength",
                       "RepetitionTime", "EchoTime", "FlipAngle", "SliceThickness", "PixelSpacing",
                       "ManufacturerModel", "StudyDate"]], width="stretch", height=300)


def pair_selector():
    pairs = pairs_df()
    if pairs.empty:
        st.warning("No complete pairs on disk yet — the download may still be running.")
        st.stop()
    sb = st.sidebar
    sb.markdown("### Choose a pair")
    split = sb.radio("Cohort", sorted(pairs.split.unique()), horizontal=True)
    p = pairs[pairs.split == split]
    if p.variant.nunique() > 1:
        p = p[p.variant == sb.radio("Variant", ["raw", "processed"], horizontal=True,
                                    help="Test cohort only: raw files vs the authors' N4/normalized/registered files")]
    plane = sb.selectbox("Plane", sorted(p.plane.unique()), format_func=PLANE_LABEL.get)
    p = p[p.plane == plane]
    fam = sb.selectbox("Sequence family", ["all"] + sorted(p.family.unique()))
    if fam != "all":
        p = p[p.family == fam]
    q = sb.text_input("Filter patient id", "")
    if q:
        p = p[p.patient.str.contains(q)]
    if p.empty:
        st.warning("No pairs match the filter.")
        st.stop()
    labels = (p.patient + "  ·  series " + p.series + "  ·  " + p.sequence).tolist()
    if "pair_idx" not in st.session_state:
        st.session_state.pair_idx = 0
    c1, c2 = sb.columns(2)
    if c1.button("◀ prev"):
        st.session_state.pair_idx -= 1
    if c2.button("random"):
        st.session_state.pair_idx = int(np.random.randint(len(p)))
    i = sb.selectbox(f"Pair ({len(p)} available)", range(len(p)), format_func=lambda k: labels[k],
                     index=st.session_state.pair_idx % len(p))
    st.session_state.pair_idx = i
    row = p.iloc[i]

    X, zx, ax_x, dt = load_vol(row.art)
    Y, zy, ax_y, _ = load_vol(row.clean)
    nz = X.shape[2]
    k = sb.slider("Slice", 0, nz - 1, nz // 2)
    # normalise both by the clean volume's 99.5th percentile (same scale for X and Y)
    s = max(np.percentile(Y, 99.5), 1e-6)
    x = to_display(X[:, :, k]) / s
    y = to_display(Y[:, :, k]) / s if Y.shape == X.shape else None
    info = dict(row=row, X=X, Y=Y, zooms=zx, axcodes=ax_x, axcodes_clean=ax_y, dtype=dt, scale=s, k=k)
    if y is None:
        st.error(f"Shape mismatch: artifact {X.shape} vs clean {Y.shape} — cannot compare voxel-wise.")
        st.stop()
    return x, y, info


def page_pairs():
    st.header("Pair explorer — corrupted X vs clean Y")
    x, y, info = pair_selector()
    row = info["row"]
    mask = y > 0.08
    st.caption(f"**{row.patient}** · {PLANE_LABEL[row.plane]} · `{row.sequence}` · {row.field_T} T · "
               f"volume {info['X'].shape} · voxel {np.round(info['zooms'], 3)} mm · orientation "
               f"{info['axcodes']} · slice {info['k']}")

    sb = st.sidebar
    sb.markdown("### Display")
    hi = sb.slider("Window max (display contrast)", 0.2, 1.5, 1.0, 0.05)
    diff_max = sb.slider("Difference colour range", 0.05, 0.6, 0.25, 0.05)
    zoom = sb.checkbox("Zoom to centre (2×)")
    if zoom:
        h, w = x.shape
        sl = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
        xv, yv = x[sl], y[sl]
    else:
        xv, yv = x, y

    fig, axs = fig_row(3, 5)
    show(axs[0], xv, "X — motion-corrupted (input)", vmin=0, vmax=hi)
    show(axs[1], yv, "Y — rescan (clean reference)", vmin=0, vmax=hi)
    show(axs[2], xv - yv, "R = X − Y (the artifact + misalignment)", cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
    st.pyplot(fig, clear_figure=True)
    explain("""
* **X** is the scan where the patient moved. Look for **ghosting** (faint shifted copies of the knee,
  usually repeating along one direction — the *phase-encode* direction), **blurred edges** and **streaks**.
* **Y** is the rescan acquired right after, when the patient kept still. It is our target.
* **R = X − Y** is the residual: red where X is brighter than Y, blue where darker. In a perfect pair R
  would contain *only* the motion artifact. In practice it also contains (a) slight **misalignment**
  between the two acquisitions (shows as thin red/blue outlines on every edge) and (b) noise.
  Ghosting appears as **repeated bands parallel to each other**; misalignment appears as **edge
  outlines that follow the anatomy**. The pair-quality check in Phase 1 tries to remove pairs where
  misalignment dominates.
* Both images are divided by the 99.5th percentile of Y, so the colour scale is identical for X and Y.
""")

    m = metrics(x, y, mask)
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k.replace("_", " "), f"{v:.3f}")
    explain("""
These numbers compare **X against Y**, i.e. how bad the corrupted image is *before* any correction —
the "uncorrected" baseline every method must beat.
* **PSNR** (dB, higher = more similar). ~30+ dB is close; the paper's U-Net reaches ~28.5 dB after correction.
* **SSIM** (0–1, higher = more similar structure, contrast and brightness in local windows).
* **NRMSE** (lower = better) — root-mean-square error relative to the image's energy.
* **MAE** — mean absolute pixel difference (in units of the normalised intensity).
* **NCC (foreground)** — correlation inside the knee only; low values (< ~0.9) suggest severe motion
  *or* misregistration.
""")

    st.subheader("Severity across all slices of this volume")
    X, Y, s = info["X"], info["Y"], info["scale"]
    per = []
    for j in range(X.shape[2]):
        xs, ys = X[:, :, j] / s, Y[:, :, j] / s
        mse = np.mean((xs - ys) ** 2)
        per.append(dict(slice=j, PSNR=10 * np.log10(1 / max(mse, 1e-12)),
                        foreground=float((ys > 0.08).mean())))
    per = pd.DataFrame(per).set_index("slice")
    c1, c2 = st.columns(2)
    c1.line_chart(per.PSNR, height=200)
    c2.bar_chart(per.foreground, height=200)
    explain("""
Left: PSNR(X, Y) per slice — dips show slices hit hardest by motion (motion affects the whole
acquisition, but some slices change more). Right: fraction of the slice covered by tissue. Slices at
the edge of the volume are nearly empty; preprocessing drops them because they inflate PSNR without
containing anatomy.
""")

    with st.expander("Raw metadata for this pair"):
        meta = load_meta()
        st.json({"artifact file": row.art.replace(str(ROOT) + "/", ""),
                 "clean file": row.clean.replace(str(ROOT) + "/", ""),
                 "shape": list(info["X"].shape), "voxel size (mm)": info["zooms"],
                 "orientation (axcodes) X / Y": [info["axcodes"], info["axcodes_clean"]],
                 "stored dtype": info["dtype"],
                 "raw intensity range X": [float(info["X"].min()), float(info["X"].max())],
                 "normalisation divisor (Y p99.5)": float(s),
                 **({k: str(v) for k, v in meta.loc[row.csv_key].items()} if row.csv_key in meta.index else {})})


def page_frequency():
    st.header("Frequency view — what motion does to the Fourier spectrum")
    x, y, info = pair_selector()
    r = x - y
    Fx, Lx = log_spec(x)
    Fy, Ly = log_spec(y)
    Fr, Lr = log_spec(r)
    vmax = np.percentile(Ly, 99.9)
    fig, axs = fig_row(4, 4)
    show(axs[0], Lx, "log|F(X)|", cmap="magma", vmin=0, vmax=vmax)
    show(axs[1], Ly, "log|F(Y)|", cmap="magma", vmin=0, vmax=vmax)
    d = Lx - Ly
    lim = np.percentile(np.abs(d), 99)
    show(axs[2], d, "ΔF = log|F(X)| − log|F(Y)|", cmap="RdBu_r", vmin=-lim, vmax=lim)
    show(axs[3], Lr, "log|F(R)| — spectrum of the artifact", cmap="magma")
    st.pyplot(fig, clear_figure=True)
    explain("""
The **2-D Fourier transform** rewrites an image as a sum of stripe patterns (sinusoids). Each point in
the spectrum is one stripe pattern: **distance from the centre = how fine** the stripes are (frequency),
**direction from the centre = which way** they run. Brightness = how much of that pattern is in the image
(we show the log because the centre is millions of times stronger than the edges).

* The **centre** holds coarse content (overall brightness and shape); the **edges** hold fine detail and noise.
* In MRI, the scanner *measures* this spectrum directly (that is **k-space**), one line at a time.
  If the knee moves between lines, those lines disagree → **ghosts** in the image. Because lines are
  collected along the **phase-encode** direction, motion corruption often shows up as **streaks /
  banding along one axis** of the spectrum.
* **ΔF** (Eq. 5 of the proposal): red = frequencies that are *stronger* in the corrupted image,
  blue = *weaker*. A horizontal or vertical red band is the fingerprint of phase-encode ghosting;
  blue at high frequencies indicates **blur** (lost fine detail).
* **log|F(R)|** is the spectrum of the residual — where the artifact's energy lives.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Radial energy profile")
        rr, ex = radial_profile(np.abs(Fx) ** 2)
        _, ey = radial_profile(np.abs(Fy) ** 2)
        _, er = radial_profile(np.abs(Fr) ** 2)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.semilogy(rr, ex, label="X (corrupted)")
        ax.semilogy(rr, ey, label="Y (clean)")
        ax.semilogy(rr, er, label="R (residual)", ls="--")
        ax.set_xlabel("normalised spatial frequency (0 = DC, 1 = Nyquist)")
        ax.set_ylabel("mean power")
        ax.legend()
        ax.grid(alpha=.3)
        st.pyplot(fig, clear_figure=True)
        explain("""
Average spectral power at each distance from the centre (each *scale*). Where the **X curve sits
above Y**, motion *added* energy at that scale (ghosts, ringing); where it sits **below**, motion
*removed* energy (blur). The **R curve** shows at which scales the artifact lives — if it overlaps the
scales where Y (real anatomy) also has energy, plain low-pass filtering cannot separate them. This is
the core motivation of the project.
""")
    with c2:
        st.subheader("Angular (directional) energy")
        th, ax_ = angular_energy(np.abs(Fx) ** 2)
        _, ay = angular_energy(np.abs(Fy) ** 2)
        _, ar = angular_energy(np.abs(Fr) ** 2)
        fig = plt.figure(figsize=(4.5, 3.8))
        ax = fig.add_subplot(projection="polar")
        for v, lab in [(ax_, "X"), (ay, "Y"), (ar, "R")]:
            ax.plot(np.r_[th, th + np.pi, th[:1]], np.r_[v, v, v[:1]], label=lab)
        ax.set_yticklabels([])
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig, clear_figure=True)
        explain("""
Share of spectral energy (excluding the very centre) falling in each direction. Real anatomy is fairly
spread out; **phase-encode motion artifacts concentrate energy along one axis**, so **R** typically
shows a pronounced lobe. If this lobe is consistent across the dataset, a *directional* representation
D(X) is a useful input for the network — exactly what Phase A will test statistically.
""")


def page_multiscale():
    st.header("Gradients, multi-scale and directional representations")
    st.caption("These are the hand-crafted DIP inputs G(X), H(X), D(X) that the proposed model receives.")
    x, y, info = pair_selector()

    st.subheader("G — gradient magnitude (Sobel)")
    gx, gy = ndi.sobel(x, 1), ndi.sobel(x, 0)
    gyx, gyy = ndi.sobel(y, 1), ndi.sobel(y, 0)
    Gx, Gy = np.hypot(gx, gy), np.hypot(gyx, gyy)
    lim = np.percentile(Gy, 99.5)
    fig, axs = fig_row(3, 4.5)
    show(axs[0], Gx, "|∇X|", cmap="inferno", vmin=0, vmax=lim)
    show(axs[1], Gy, "|∇Y|", cmap="inferno", vmin=0, vmax=lim)
    show(axs[2], Gx - Gy, "|∇X| − |∇Y|", cmap="RdBu_r", vmin=-lim / 2, vmax=lim / 2)
    st.pyplot(fig, clear_figure=True)
    explain("""
The **gradient** measures how fast brightness changes — it lights up **edges** (cartilage surface,
bone boundary, meniscus). **Sobel** is a small 3×3 filter that estimates it. Red in the difference =
edges *present in X but not in Y* (ghost edges); blue = real edges *weakened* by motion blur.
A good restoration must remove the red without deleting the blue.
""")

    st.subheader("H — Laplacian pyramid (multi-scale bands)")
    bx, by = laplacian_pyramid(x), laplacian_pyramid(y)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
    for j in range(4):
        lab = f"band {j} (finest)" if j == 0 else ("low-pass residual L" if j == 3 else f"band {j}")
        b = np.percentile(np.abs(by[j]), 99) if j < 3 else None
        show(axs[0, j], bx[j], f"X · {lab}", vmin=-b if b else None, vmax=b)
        show(axs[1, j], by[j], f"Y · {lab}", vmin=-b if b else None, vmax=b)
    st.pyplot(fig, clear_figure=True)
    explain("""
A **Laplacian pyramid** splits an image into **bands of detail at different sizes**: band 0 holds the
finest detail (pixel-level texture and noise), band 1 slightly coarser structures, and so on, with the
last image L holding the smooth overall shape. Adding all bands back gives the original exactly
(Eq. 6: X → {L, H1, H2, …}). Compare rows: motion ghosts often live in the **middle bands**, while
noise lives in band 0 — this tells us which scales the artifact occupies.
""")

    st.subheader("Wavelet sub-bands (Haar, level 1)")
    cA, (cH, cV, cD) = pywt.dwt2(x, "haar")
    cAy, (cHy, cVy, cDy) = pywt.dwt2(y, "haar")
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.6))
    for j, (a, b, n) in enumerate([(cH, cHy, "horizontal detail (LH)"), (cV, cVy, "vertical detail (HL)"),
                                   (cD, cDy, "diagonal detail (HH)")]):
        lim = np.percentile(np.abs(b), 99)
        show(axs[0, j], a, f"X · {n}", vmin=-lim, vmax=lim)
        show(axs[1, j], b, f"Y · {n}", vmin=-lim, vmax=lim)
    st.pyplot(fig, clear_figure=True)
    e = lambda a: float(np.mean(a ** 2))  # noqa: E731
    st.dataframe(pd.DataFrame({"X energy": [e(cH), e(cV), e(cD)], "Y energy": [e(cHy), e(cVy), e(cDy)]},
                              index=["LH (horizontal edges)", "HL (vertical edges)", "HH (diagonal)"])
                 .assign(ratio_X_over_Y=lambda d: d["X energy"] / d["Y energy"]))
    explain("""
A **wavelet transform** is like the pyramid but also separates **direction**: LH responds to
horizontal edges, HL to vertical edges, HH to diagonals. If the energy ratio X/Y is much larger in
one sub-band than the others, motion adds structure in that orientation — a direct, cheap test of
whether the artifact is directional.
""")

    st.subheader("D — Gabor directional responses")
    gb_x, gb_y = gabor_bank(x), gabor_bank(y)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
    lim = max(np.percentile(r, 99.5) for _, r in gb_y)
    for j in range(4):
        show(axs[0, j], gb_x[j][1], f"X · θ = {gb_x[j][0]:.0f}°", cmap="viridis", vmin=0, vmax=lim)
        show(axs[1, j], gb_y[j][1], f"Y · θ = {gb_y[j][0]:.0f}°", cmap="viridis", vmin=0, vmax=lim)
    st.pyplot(fig, clear_figure=True)
    explain("""
A **Gabor filter** is a small striped patch (a sinusoid inside a Gaussian window) tuned to **one
orientation and one scale**. Convolving the image with it lights up structures at that orientation.
We use 4 orientations (0°, 45°, 90°, 135°). If X lights up much more than Y at one particular angle
across many pairs, motion artifacts are orientation-specific — and D(X) gives the network an explicit
handle on them (Section 5.2 of the proposal).
""")


def page_glossary():
    st.header("Glossary — every technical term in this project")
    q = st.text_input("Search", "")
    for section, items in GLOSSARY.items():
        hits = [(t, d) for t, d in items if not q or q.lower() in (t + d).lower()]
        if not hits:
            continue
        st.subheader(section)
        for term, desc in hits:
            with st.expander(term, expanded=bool(q)):
                st.markdown(desc)


PAGES = {"Overview": page_overview, "Pair explorer": page_pairs, "Frequency view": page_frequency,
         "Gradients / multi-scale / directional": page_multiscale, "Glossary": page_glossary}
st.sidebar.title("KMAR-50K Viewer")
choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice]()
