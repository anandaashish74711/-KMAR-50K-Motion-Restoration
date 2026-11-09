"""Build data/inventory.csv: one row per (artifact, clean) pair.

Columns: split, patient, plane, series, variant, sequence (SeriesDescription),
field strength, shapes, spacing, intensity stats, paths.

File naming (KMAR-50K):
    <patient>_<plane>_<series>.nii.gz            e.g. 2020_100_sagittal_0.0.nii.gz
    test cohort also has preprocessed variants:
    <...>_N4_Norm.nii.gz (artifact), <...>_N4_Norm_RegSyN.nii.gz (clean)
"""
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kmar50k"
META = ROOT / "data" / "raw" / "meta"
NAME_RE = re.compile(r"^(?P<patient>.+?)_(?P<plane>sagittal|coronal|transection)_(?P<series>\d+(?:\.\d+)?)"
                     r"(?P<suffix>_N4_Norm(?:_RegSyN)?)?\.nii(?:-\w+)?\.gz$")


def parse(fn):
    m = NAME_RE.match(fn)
    if not m:
        return None
    d = m.groupdict()
    d["variant"] = "processed" if d["suffix"] else "raw"
    # canonical pair key ignores the preprocessing suffix (artifact=_N4_Norm, clean=_N4_Norm_RegSyN)
    d["key"] = f"{d['patient']}_{d['plane']}_{d['series']}_{d['variant']}"
    d["csv_key"] = f"{d['patient']}_{d['plane']}_{int(float(d['series']))}"
    return d


def load_cohort_meta():
    rows = []
    for fn, col in [("TrainingCohort.csv", "NII_FileName"), ("TestingCohort.csv", "Patient")]:
        df = pd.read_csv(META / fn)
        df["csv_key"] = df[col].astype(str).str.replace(".nii.gz", "", regex=False).str.strip()
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    clean = lambda s: str(s).strip().strip("'").strip()  # noqa: E731
    out = pd.DataFrame({
        "csv_key": df["csv_key"],
        "sequence": df["SeriesDescription"].map(clean),
        "field_T": df["MagneticFieldStrength"].map(clean),
        "slice_thickness": df["SliceThickness"].map(clean),
        "pixel_spacing_csv": df["PixelSpacing"].map(clean),
        "scanner": df["ManufacturerModel"].map(clean),
    })
    return out.drop_duplicates("csv_key")


def header_info(path):
    img = nib.load(str(path))
    a = np.asarray(img.dataobj, dtype=np.float32)
    return dict(shape="x".join(map(str, a.shape)),
                zooms="x".join(f"{z:.3f}" for z in img.header.get_zooms()[:3]),
                vmin=float(a.min()), vmax=float(a.max()), vmean=float(a.mean()),
                axcodes="".join(nib.aff2axcodes(img.affine)))


def main():
    pairs, unmatched = [], []
    for split in ["train", "test"]:
        files = {}
        for kind in ["artifact", "clean"]:
            for p in sorted((RAW / split / kind).glob("*.gz")):
                d = parse(p.name)
                if d is None:
                    unmatched.append((split, kind, p.name, "unparsable"))
                    continue
                files.setdefault(d["key"], {})[kind] = (p, d)
        for key, kinds in sorted(files.items()):
            if set(kinds) != {"artifact", "clean"}:
                for k, (p, _) in kinds.items():
                    unmatched.append((split, k, p.name, "no partner"))
                continue
            (pa, d), (pc, _) = kinds["artifact"], kinds["clean"]
            ha, hc = header_info(pa), header_info(pc)
            pairs.append(dict(split=split, patient=d["patient"], plane=d["plane"], series=d["series"],
                              variant=d["variant"], csv_key=d["csv_key"],
                              shape_art=ha["shape"], shape_clean=hc["shape"],
                              zooms_art=ha["zooms"], zooms_clean=hc["zooms"],
                              axcodes_art=ha["axcodes"], axcodes_clean=hc["axcodes"],
                              range_art=f"{ha['vmin']:.3g}..{ha['vmax']:.3g}",
                              range_clean=f"{hc['vmin']:.3g}..{hc['vmax']:.3g}",
                              mean_art=ha["vmean"], mean_clean=hc["vmean"],
                              path_art=str(pa.relative_to(ROOT)), path_clean=str(pc.relative_to(ROOT))))
        print(f"{split}: {sum(p['split'] == split for p in pairs)} pairs", flush=True)

    inv = pd.DataFrame(pairs).merge(load_cohort_meta(), on="csv_key", how="left")
    inv["same_shape"] = inv.shape_art == inv.shape_clean
    inv.to_csv(ROOT / "data" / "inventory.csv", index=False)
    pd.DataFrame(unmatched, columns=["split", "kind", "file", "reason"]).to_csv(
        ROOT / "data" / "inventory_unmatched.csv", index=False)

    print(f"\n{len(inv)} pairs, {len(unmatched)} unmatched files")
    print("\nPairs by split / variant / plane:")
    print(inv.groupby(["split", "variant", "plane"]).size().unstack(fill_value=0))
    seq = inv.sequence.fillna("?").str.lower().str.extract(r"^(q?(?:pd|t1|t2)[a-z_]*?_(?:tra|sag|cor))")[0]
    print("\nSequence family x plane (train, raw):")
    m = (inv.split == "train") & (inv.variant == "raw")
    print(pd.crosstab(seq[m].fillna("other"), inv.plane[m]))
    print("\nShape mismatches:", int((~inv.same_shape).sum()))
    print("Missing CSV metadata:", int(inv.sequence.isna().sum()))
    print("Patients (train/test):", inv[inv.split == "train"].patient.nunique(),
          inv[inv.split == "test"].patient.nunique())
    print("Patient overlap train/test:", len(set(inv[inv.split == "train"].patient) & set(inv[inv.split == "test"].patient)))


if __name__ == "__main__":
    sys.exit(main())
