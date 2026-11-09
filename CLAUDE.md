# DIP Project — notes for Claude sessions

Course project: frequency-guided motion artifact reduction in knee MRI (KMAR-50K). Read PLAN.md first; it is the source of truth for dataset handling, models (M1 U-Net, M2 residual U-Net, M3 FG-ResUNet with [X,G,H,D] input), baselines, ablations and metrics.

## Rules
- Work phase by phase (PLAN.md §9); the user reviews each phase before the next.
- Splits are **patient-level**, saved once to data/splits.json — never regenerate or reshuffle.
- Phase A analysis and all tuning use train/val only; test is touched only for final evaluation.
- No vertical flips / 90° rotations in augmentation (would scramble phase-encode-direction ghosting).
- All learned models share the same backbone and iteration budget; only the input stem differs.

## Environment
- venv: /archive/simran.singh/envs/dip (conda is broken on this machine — don't use it).
- GPU: check `nvidia-smi` first; GPU 1 was free when the project started. Use CUDA_VISIBLE_DEVICES.
- data/raw, data/processed, runs/ are git-ignored (large).

## Status
- [ ] Phase 1: setup, download, inventory, subset choice, pair-quality check, preprocess, splits
- [ ] Phase 2: Phase A spectral/directional analysis
- [ ] Phase 3: classical baselines
- [ ] Phase 4: M1, M2
- [ ] Phase 5: M3 + ablations
- [ ] Phase 6: final evaluation, figures, tables
