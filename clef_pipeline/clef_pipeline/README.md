# CLEF Pipeline — κ_λ Extraction from pp → HH → bbbb

## Framework
**CLEF** (*Constantes et Lois Émergentes du Fondamental*)  
Prediction: **κ_λ = 1.5738738045** (Higgs trilinear self-coupling modifier)  
Derived via Coleman-Weinberg potential with fractal dimension d_f = 11/3 and IR graviton mass cutoff.

## Pipeline Architecture

```
MadGraph5_aMC@NLO          Pythia8              Delphes
┌──────────────┐       ┌──────────────┐     ┌──────────────┐
│ gg → HH (NLO)│──LHE──│ PS + hadron. │─HepMC─│  CMS HL-LHC  │
│ + MadSpin    │       │ Monash 2013  │     │  b-tagging   │
│ + reweight   │       │ MPI + CR     │     │  anti-kT 0.4 │
└──────────────┘       └──────────────┘     └──────┬───────┘
                                                    │ ROOT
                                              ┌─────▼──────┐
                                              │ root_to_part│
                                              │ χ² pairing  │
                                              └─────┬───────┘
                                                    │ HDF5
                                        ┌───────────▼──────────┐
                                        │ Particle Transformer │
                                        │  6-layer, 8-head     │
                                        │  pair-aware attention │
                                        │  dual head:          │
                                        │   • classification   │
                                        │   • κ_λ regression   │
                                        └───────────┬──────────┘
                                                    │
                                        ┌───────────▼──────────┐
                                        │  Profile Likelihood  │
                                        │  m_HH morphing       │
                                        │  -2ΔlnL scan         │
                                        │  CLEF vs SM test     │
                                        └──────────────────────┘
```

## Directory Structure

```
clef_pipeline/
├── madgraph/
│   ├── proc_card_HH_bbbb.dat    # Process definition
│   ├── run_card_HH_bbbb.dat     # Run configuration (14 TeV)
│   ├── param_card_CLEF.dat      # SM params + κ_λ = 1.5738738045
│   ├── madspin_card.dat         # H → bb decay
│   └── reweight_card.dat        # κ_λ scan points
├── pythia8/
│   └── pythia8_HH_bbbb.cmnd    # Shower configuration
├── delphes/
│   └── delphes_card_HLLHC_bbbb.tcl  # CMS HL-LHC detector
├── part/
│   └── part_model.py            # Particle Transformer
├── analysis/
│   ├── root_to_part.py          # ROOT → HDF5 converter
│   └── extract_kappa_lambda.py  # κ_λ profile likelihood
├── scripts/
│   └── run_pipeline.sh          # Master orchestration
├── data/
│   ├── signal/                  # MG5/Pythia/Delphes output
│   ├── background/              # QCD 4b, ttbar
│   └── processed/               # HDF5 for ParT
└── results/                     # Final outputs
```

## Quick Start

```bash
# Full pipeline
./scripts/run_pipeline.sh all

# Or step by step
./scripts/run_pipeline.sh madgraph
./scripts/run_pipeline.sh shower
./scripts/run_pipeline.sh delphes
./scripts/run_pipeline.sh convert
./scripts/run_pipeline.sh train
./scripts/run_pipeline.sh extract
```

## Physics

### Signal Process
- **Process**: gg → HH → bb̄bb̄ (loop-induced, NLO QCD)
- **σ(SM)**: 31.05 fb × BR(bbbb) = 10.5 fb
- **σ(CLEF, κ_λ=1.5738738045)**: ~8.8 fb (reduced due to enhanced destructive interference)

### Key Observables
- **m_HH**: Primary discriminant (shape depends on κ_λ)
- **m_H1, m_H2**: Higgs candidate masses (χ² pairing)
- **ΔR(H1), ΔR(H2)**: Angular separation of b-jets in each Higgs
- **ParT score**: Signal/background classifier output

### Expected Sensitivity (HL-LHC, 3 ab⁻¹)
- **σ(κ_λ)** ≈ 0.3–0.5 (depending on systematics)
- **CLEF vs SM discrimination**: ~1.5–2.5σ in bbbb channel alone
- Combined with bbττ, bbγγ: potentially 3–4σ

## Prerequisites

- MadGraph5_aMC@NLO ≥ 3.5.0
- Pythia8 ≥ 8.310
- Delphes ≥ 3.5.0
- LHAPDF6 with NNPDF31_nnlo_as_0118
- Python 3.9+ with: torch, uproot, awkward, h5py, scipy, scikit-learn
- Optional: matplotlib (plots), ROOT (direct analysis)

## κ_λ Reweighting

The reweight card generates event weights for 11 κ_λ hypotheses from a single
MC sample, using the analytic decomposition:

  σ(κ_λ) = A₁ + κ_λ·A₃ + κ_λ²·A₇

where A₁ (box), A₃ (interference), A₇ (triangle) are basis amplitudes.

## References

- Qu & Gouskos, "Particle Transformer" (arXiv:2202.03772)
- Grazzini et al., HH NLO cross sections (arXiv:1803.02463)
- CMS HH→bbbb HL-LHC projection (CMS-PAS-FTR-18-019)
