# Fe6Co2Ge Result Summary

Date: 2026-03-27

This note freezes the current first-pass result for `Fe6Co2Ge` from the `GNoME -> GNN -> DFT` pipeline.

## Structure

- Formula: `Fe6Co2Ge`
- Screened structure id: `9d7cf9b43f`
- Crystal system / space group: trigonal `P3m1`
- Atoms per cell: `9`
- Cell parameters from `validation/fe6co2ge_qe/fe6co2ge_gnome.cif`:
  - `a = 4.026667 Å`
  - `c = 7.461256 Å`
  - `V = 104.7692907414 Å^3`

## GNN Prediction

Source:
- `runs/gnome_screen_full_clean_v2_uncertainty_ordering_merged/top30_low_cost.csv`

Prediction for `Fe6Co2Ge`:
- predicted ordering: `FM`
- `p_FM = 0.999968`
- magnetic probability: `0.999997`
- predicted moment: `1.983643 μB/atom`
- predicted transition temperature: `459.146729 K`
- predicted energy above hull: `0.022378 eV/atom`
- predicted formation energy: `-0.115793 eV/atom`
- predicted band gap: `5e-06 eV`

Interpretation:
- Strong magnetic hit.
- Rare-earth-free.
- Predicted to be metallic and reasonably low above hull.

## Easy DFT Validation

Source:
- `validation/DFT_validation_summary.md`

QE validation summary:
- `FM`: converged
- `FM` energy: `-2705.30954958 Ry`
- `FM` total magnetization: `17.30 μB/cell`
- `FM` moment scale: about `1.92 μB/atom`
- `NM`: converged
- `NM` energy: `-2705.02977892 Ry`
- `FiM`: finished without convergence
- `FiM` last energy: `-2705.14807981 Ry`

Energy differences:
- `NM - FM = 422.94 meV/atom`
- non-converged `FiM - FM ≈ 244.72 meV/atom`

Interpretation:
- The simple DFT confirms `FM` is strongly favored.
- The GNN got the magnetic ground state right at the easy-validation level.

## Force-Theorem MAE Result

Method:
- Scalar FM seed SCF
- SOC NSCF band-energy calculations for `[001]` and `[100]`
- MAE evaluated as `E[100] - E[001]`

Final completed outputs on cluster:
- `~/Edo/Magnets/validation/fe6co2ge_mae_qe/ft_nscf_100_resume4.out`
- `~/Edo/Magnets/validation/fe6co2ge_mae_qe/ft_nscf_001_long.out`

Exact output lines:
- `[100]`: `eband, Ef (eV) = -1877.5428032682614  17.862815705629981`
- `[001]`: `eband, Ef (eV) = -1877.5532175695848  17.862726985411218`
- both outputs end with `JOB DONE.`

Computed MAE:
- `E[100] - E[001] = +0.010414301323407926 eV/cell`
- `= +10.414301323407926 meV/cell`
- Conversion factor for this cell: `1 meV/cell = 1.5292426079 MJ/m^3`
- `MAE ≈ +15.9259933151 MJ/m^3`

Interpretation:
- Sign is favorable.
- `[001]` is lower than `[100]`.
- The easy axis is along `[001]`.
- The anisotropy difference is extremely large on this first-pass calculation.

## Current Bottom Line

What is supported now:
- `Fe6Co2Ge` is a strong ferromagnetic candidate.
- The easy DFT validation supports `FM` clearly over `NM`.
- The first-pass SOC force-theorem result predicts a very large positive anisotropy with easy axis `[001]`.

Safe wording:
- `Fe6Co2Ge` is a strong rare-earth-free permanent-magnet candidate pending tighter confirmation.

Important caution:
- This MAE number is a first-pass force-theorem scout result for an ideal crystal.
- It is exciting enough to justify follow-up, but it should not yet be presented as a final publication-grade anisotropy constant without tighter verification.
