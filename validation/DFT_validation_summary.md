# DFT Validation Summary

This note summarizes the quick Quantum ESPRESSO validation runs completed so far for top screened GNoME candidates.

These are **fast SCF validation runs**, not full publication-grade workflows.

- Code: `Quantum ESPRESSO 7.4.1`
- Execution style: `2 MPI ranks`, `CG` diagonalization
- Typical settings: `ecutwfc = 60 Ry`, `ecutrho = 480 Ry`, Marzari-Vanderbilt smearing `0.04 Ry`
- Purpose: check whether the model gets the **magnetic ground state** and **moment scale** roughly right
- Not included yet: structural relaxation, SOC, anisotropy, final high-accuracy convergence study

Raw outputs are available locally in [qe_smoke](/Users/edo/Edo/Magnets/validation/qe_smoke).

## Fe2Co

Model prediction on the exact screened GNoME structure:

- `material_id`: `95369b14fd`
- structure: `P-3m1`, `6` atoms
- predicted ordering: `FM`
- predicted moment: `2.560863 μB/atom`
- predicted `Tc/TN`: `432.99 K`
- predicted `E_hull`: `0.026441 eV/atom`
- predicted band gap: `0.001111 eV`

DFT results:

- `FM`: `-1890.39383018 Ry`
- `FiM`: `-1890.28759647 Ry`
- `NM`: `-1890.14771055 Ry`
- ordering by energy: `FM < FiM < NM`
- `FiM - FM`: `240.90 meV/atom`
- `NM - FM`: `558.10 meV/atom`
- final `FM` total magnetization: `14.01 μB/cell`
- final `FM` moment scale: `2.335 μB/atom`

Interpretation:

- Strong validation win.
- The model correctly predicted `FM`.
- The predicted moment was slightly high, but in the right ballpark.

Files:

- [fe2co_fm_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/fe2co_fm_2p.out)
- [fe2co_fim_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/fe2co_fim_2p.out)
- [fe2co_nm_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/fe2co_nm_2p.out)

## VFe15

Model prediction on the exact screened GNoME structure:

- `material_id`: `1c3630d10b`
- structure: `Pm-3m`, `16` atoms
- predicted ordering: `FiM`
- predicted moment: `2.286992 μB/atom`
- predicted `Tc/TN`: `465.82 K`
- predicted `E_hull`: `0.018117 eV/atom`
- predicted band gap: `0.000007 eV`

DFT results:

- `FM`: `-5115.56110488 Ry`
- `FiM`: `-5115.56110071 Ry`
- `NM`: `-5115.10607292 Ry`
- `FM - FiM`: `0.0035 meV/atom` in favor of `FM`
- `NM - FM`: `386.94 meV/atom`
- final magnetic moment scale: about `30.82-30.83 μB/cell`
- final moment scale per atom: about `1.93 μB/atom`

Important local-moment result:

- In both the `FM`-initialized and `FiM`-initialized runs, the `V` site converged to a **negative** local moment:
  - `FM init`: about `-1.0604 μB`
  - `FiM init`: about `-1.0711 μB`
- All `Fe` sites stayed **positive** with moments around `+2.13` to `+2.52 μB`

Interpretation:

- `NM` is decisively ruled out.
- `FM` and `FiM` are effectively degenerate at this validation level.
- The physically meaningful local-moment pattern is ferrimagnetic in both magnetic runs.
- So although the final total energies differ by a tiny amount in favor of `FM`, the model's `FiM` prediction is still physically well supported.

Files:

- [vfe15_fm_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/vfe15_fm_2p.out)
- [vfe15_fim_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/vfe15_fim_2p.out)
- [vfe15_nm_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/vfe15_nm_2p.out)

## Fe6Co2Ge

Model prediction on the exact screened GNoME structure:

- `material_id`: `9d7cf9b43f`
- structure: `P3m1`, `9` atoms
- predicted ordering: `FM`
- predicted moment: `1.983643 μB/atom`
- predicted `Tc/TN`: `459.15 K`
- predicted `E_hull`: `0.022378 eV/atom`
- predicted band gap: `0.000005 eV`

Current DFT status:

- `FM`: **converged**
  - energy: `-2705.30954958 Ry`
  - total magnetization: `17.30 μB/cell`
  - moment scale: about `1.92 μB/atom`
- `NM`: **converged**
  - energy: `-2705.02977892 Ry`
- `FiM`: **finished without convergence**
  - stopped after `100` SCF iterations
  - last energy: `-2705.14807981 Ry`
  - last total magnetization: `-3.20 μB/cell`

Current energy comparison:

- `NM - FM`: `422.94 meV/atom`
- non-converged `FiM - FM`: about `244.72 meV/atom`

Interpretation:

- This is already a strong early validation result.
- `FM` is clearly below `NM`.
- `FM` is also far below the non-converged `FiM` run.
- Even though the `FiM` branch did not converge cleanly, the result is still strongly consistent with `FM` being the magnetic ground state.

Files:

- [fe6co2ge_fm_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/fe6co2ge_fm_2p.out)
- [fe6co2ge_fim_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/fe6co2ge_fim_2p.out)
- [fe6co2ge_nm_2p.out](/Users/edo/Edo/Magnets/validation/qe_smoke/fe6co2ge_nm_2p.out)

## Overall Read

So far the model is doing well on the magnetic ground-state problem:

- `Fe2Co`: clear `FM` match
- `VFe15`: strongly magnetic, with `FM/FiM` near-degeneracy and a ferrimagnetic local-moment pattern
- `Fe6Co2Ge`: currently looks like another strong `FM` match

The strongest robust conclusion across all three is:

- the model is very good at identifying when a candidate is **strongly magnetic rather than non-magnetic**
- the fine distinction between `FM` and `FiM` can be subtle, but even there the model has been physically sensible

## Next Step: Fe6Co2Ge MAE

The next follow-up is now set up for `Fe6Co2Ge`:

- first-pass SOC / noncollinear MAE check
- direction 1: `[001]` / `c` axis
- direction 2: in-plane (`[100]`-type)
- goal: estimate whether the material shows meaningful uniaxial anisotropy

Inputs:

- [mae_001.in](/Users/edo/Edo/Magnets/validation/fe6co2ge_mae_qe/mae_001.in)
- [mae_100.in](/Users/edo/Edo/Magnets/validation/fe6co2ge_mae_qe/mae_100.in)

This is a first-pass MAE screen, not yet a final fully converged publication-grade anisotropy workflow.
