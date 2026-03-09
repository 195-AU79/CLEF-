#!/bin/bash
#=================================================================
#  CLEF Pipeline — Master Orchestration Script
#  pp → HH → bbbb @ 14 TeV
#  κ_λ = 1.5738738045 (CLEF prediction)
#=================================================================
#
#  Chain: MadGraph5 → Pythia8 → Delphes → ROOT→h5 → ParT → κ_λ
#
#  Prerequisites:
#    - MadGraph5_aMC@NLO (v3.5+)
#    - Pythia8 (v8.3+)
#    - Delphes (v3.5+)
#    - Python 3.9+ with: torch, uproot, awkward, h5py, scipy
#    - LHAPDF6 with NNPDF31_nnlo_as_0118
#
#  Usage:
#    ./run_pipeline.sh [step]
#    Steps: all | madgraph | shower | delphes | convert | train | extract
#=================================================================

set -euo pipefail

# ========================= CONFIGURATION =========================
export PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MG5_DIR="${MG5_DIR:-/opt/MadGraph5}"
export DELPHES_DIR="${DELPHES_DIR:-/opt/Delphes}"
export PYTHIA8_DIR="${PYTHIA8_DIR:-/opt/pythia8}"

# Physics parameters
export KL_CLEF=1.5738738045       # CLEF prediction
export KL_SM=1.0           # SM value
export SQRT_S=14000        # 14 TeV
export LUMINOSITY=3000     # 3 ab⁻¹ (HL-LHC)
export N_EVENTS=100000     # Events per sample

# Paths
CARDS_DIR="${PIPELINE_DIR}/madgraph"
PYTHIA_CFG="${PIPELINE_DIR}/pythia8/pythia8_HH_bbbb.cmnd"
DELPHES_CARD="${PIPELINE_DIR}/delphes/delphes_card_HLLHC_bbbb.tcl"
DATA_DIR="${PIPELINE_DIR}/data"
RESULTS_DIR="${PIPELINE_DIR}/results"

mkdir -p "${DATA_DIR}/signal" "${DATA_DIR}/background" "${DATA_DIR}/processed"
mkdir -p "${RESULTS_DIR}"

STEP="${1:-all}"

echo "============================================================"
echo "  CLEF Pipeline — pp → HH → bbbb @ 14 TeV"
echo "  κ_λ(CLEF) = ${KL_CLEF}"
echo "  √s = ${SQRT_S} GeV, L = ${LUMINOSITY} fb⁻¹"
echo "  Step: ${STEP}"
echo "============================================================"
echo ""

# ========================= STEP 1: MADGRAPH5 =========================
run_madgraph() {
    echo ">>> STEP 1: MadGraph5 — Generating pp → HH events"
    echo "    Process: gg → HH (loop-induced, NLO)"
    echo "    N_events: ${N_EVENTS}"
    echo ""
    
    cd "${DATA_DIR}/signal"
    
    # Generate signal at SM (κ_λ=1), then reweight to CLEF
    ${MG5_DIR}/bin/mg5_aMC << EOF
import model loop_sm-no_b_mass
define p = g u c d s u~ c~ d~ s~ b b~
generate g g > h h [noborn=QCD]
output pp_HH_NLO
launch pp_HH_NLO
  set nevents ${N_EVENTS}
  set ebeam1 7000
  set ebeam2 7000
  set pdlabel lhapdf
  set lhaid 303600
  set dynamical_scale_choice 1
  set use_syst True
  # Decay via MadSpin
  madspin=ON
  decay h > b b~
  decay h > b b~
  # Reweight for κ_λ scan
  reweight=ON
  done
EOF

    echo "    MadGraph5 complete. LHE file at: pp_HH_NLO/Events/"
    
    # Also generate backgrounds
    echo ">>> Generating QCD 4b background..."
    ${MG5_DIR}/bin/mg5_aMC << EOF
import model sm
define p = g u c d s u~ c~ d~ s~ b b~
generate p p > b b~ b b~
output pp_4b
launch pp_4b
  set nevents ${N_EVENTS}
  set ebeam1 7000
  set ebeam2 7000
  set ptb 20
  set etab 5
  done
EOF

    echo ">>> Generating tt̄ background..."
    ${MG5_DIR}/bin/mg5_aMC << EOF
import model sm
generate p p > t t~, (t > w+ b, w+ > j j), (t~ > w- b~, w- > j j)
output pp_ttbar
launch pp_ttbar
  set nevents ${N_EVENTS}
  set ebeam1 7000
  set ebeam2 7000
  done
EOF

    echo "    All MadGraph5 generation complete."
}


# ========================= STEP 2: PYTHIA8 SHOWER =========================
run_shower() {
    echo ">>> STEP 2: Pythia8 — Parton shower + hadronization"
    
    for proc in pp_HH_NLO pp_4b pp_ttbar; do
        echo "    Showering: ${proc}"
        
        LHE_FILE="${DATA_DIR}/signal/${proc}/Events/run_01/unweighted_events.lhe.gz"
        if [ ! -f "$LHE_FILE" ]; then
            LHE_FILE="${DATA_DIR}/signal/${proc}/Events/run_01/events.lhe.gz"
        fi
        
        HEPMC_FILE="${DATA_DIR}/signal/${proc}_showered.hepmc"
        
        # Run Pythia8 with MG5 interface (or standalone)
        ${PYTHIA8_DIR}/examples/main89 \
            --config "${PYTHIA_CFG}" \
            --input "${LHE_FILE}" \
            --output "${HEPMC_FILE}" \
            --nevents ${N_EVENTS}
        
        echo "    → ${HEPMC_FILE}"
    done
    
    echo "    Pythia8 showering complete."
}


# ========================= STEP 3: DELPHES =========================
run_delphes() {
    echo ">>> STEP 3: Delphes — Detector simulation"
    
    for proc in pp_HH_NLO pp_4b pp_ttbar; do
        echo "    Simulating: ${proc}"
        
        HEPMC_FILE="${DATA_DIR}/signal/${proc}_showered.hepmc"
        ROOT_FILE="${DATA_DIR}/signal/${proc}_delphes.root"
        
        ${DELPHES_DIR}/DelphesHepMC2 \
            "${DELPHES_CARD}" \
            "${ROOT_FILE}" \
            "${HEPMC_FILE}"
        
        echo "    → ${ROOT_FILE}"
    done
    
    echo "    Delphes simulation complete."
}


# ========================= STEP 4: CONVERT ROOT → H5 =========================
run_convert() {
    echo ">>> STEP 4: ROOT → HDF5 conversion for ParT"
    
    # Signal (κ_λ = 1.5738738045 via reweighting)
    python3 "${PIPELINE_DIR}/analysis/root_to_part.py" \
        --input "${DATA_DIR}/signal/pp_HH_NLO_delphes.root" \
        --output "${DATA_DIR}/processed/signal_kl1p5739.h5" \
        --label 1 \
        --kl ${KL_CLEF}
    
    # SM signal (κ_λ = 1.0)
    python3 "${PIPELINE_DIR}/analysis/root_to_part.py" \
        --input "${DATA_DIR}/signal/pp_HH_NLO_delphes.root" \
        --output "${DATA_DIR}/processed/signal_kl1p000.h5" \
        --label 1 \
        --kl ${KL_SM}
    
    # QCD 4b background
    python3 "${PIPELINE_DIR}/analysis/root_to_part.py" \
        --input "${DATA_DIR}/signal/pp_4b_delphes.root" \
        --output "${DATA_DIR}/processed/bkg_qcd4b.h5" \
        --label 0 \
        --kl 0
    
    # tt̄ background
    python3 "${PIPELINE_DIR}/analysis/root_to_part.py" \
        --input "${DATA_DIR}/signal/pp_ttbar_delphes.root" \
        --output "${DATA_DIR}/processed/bkg_ttbar.h5" \
        --label 0 \
        --kl 0
    
    echo "    Conversion complete. Files in ${DATA_DIR}/processed/"
}


# ========================= STEP 5: TRAIN PART =========================
run_train() {
    echo ">>> STEP 5: Training Particle Transformer"
    
    python3 "${PIPELINE_DIR}/part/part_model.py" \
        --mode train \
        --data-dir "${DATA_DIR}/processed/" \
        --output "${RESULTS_DIR}/" \
        --epochs 50
    
    echo "    ParT training complete."
}


# ========================= STEP 6: EXTRACT κ_λ =========================
run_extract() {
    echo ">>> STEP 6: κ_λ extraction via profile likelihood"
    
    python3 "${PIPELINE_DIR}/analysis/extract_kappa_lambda.py" \
        --luminosity ${LUMINOSITY} \
        --kl-true ${KL_CLEF} \
        --efficiency 0.40 \
        --output "${RESULTS_DIR}/" \
        --plot
    
    echo "    κ_λ extraction complete."
    echo ""
    echo "============================================================"
    echo "  PIPELINE COMPLETE"
    echo "  Results in: ${RESULTS_DIR}/"
    echo "============================================================"
}


# ========================= DISPATCH =========================
case "${STEP}" in
    all)
        run_madgraph
        run_shower
        run_delphes
        run_convert
        run_train
        run_extract
        ;;
    madgraph)  run_madgraph ;;
    shower)    run_shower ;;
    delphes)   run_delphes ;;
    convert)   run_convert ;;
    train)     run_train ;;
    extract)   run_extract ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: $0 [all|madgraph|shower|delphes|convert|train|extract]"
        exit 1
        ;;
esac
