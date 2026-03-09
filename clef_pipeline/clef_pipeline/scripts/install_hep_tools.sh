#!/bin/bash
#=================================================================
#  CLEF Pipeline — Installation des outils HEP
#  MadGraph5_aMC@NLO + Pythia8 + Delphes + LHAPDF
#
#  Testé sur: Ubuntu 22.04/24.04, macOS (avec Homebrew)
#  Temps estimé: ~30-45 min (compilation incluse)
#=================================================================

set -euo pipefail

# ========================= PRÉREQUIS SYSTÈME =========================
echo ">>> Étape 0: Installation des prérequis système"

# --- Ubuntu/Debian ---
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    gfortran \
    gcc \
    g++ \
    cmake \
    wget \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    zlib1g-dev \
    libboost-all-dev \
    libbz2-dev \
    rsync \
    tcl \
    ghostscript \
    texlive-base

# --- macOS (Homebrew) ---
# brew install gcc gfortran cmake wget boost zlib python@3.11 tcl-tk

# Créer le répertoire d'installation
export HEP_DIR="${HOME}/hep_tools"
mkdir -p ${HEP_DIR}
cd ${HEP_DIR}

echo ""
echo "============================================================"
echo "  Tout sera installé dans: ${HEP_DIR}"
echo "============================================================"
echo ""


# ========================= 1. ROOT (CERN) =========================
# ROOT est nécessaire pour Delphes. Option rapide: binaire pré-compilé.
echo ">>> Étape 1: ROOT (CERN)"

# Option A: Conda (le plus simple)
# conda install -c conda-forge root

# Option B: Binaire pré-compilé
ROOT_VERSION="6.32.02"
wget -q https://root.cern/download/root_v${ROOT_VERSION}.Linux-ubuntu24.04-x86_64-gcc13.2.tar.gz \
    -O root.tar.gz
tar xzf root.tar.gz
rm root.tar.gz
source ${HEP_DIR}/root/bin/thisroot.sh

echo "  ROOT installé: $(root-config --version)"


# ========================= 2. LHAPDF6 =========================
echo ">>> Étape 2: LHAPDF6 (bibliothèque PDF)"

LHAPDF_VERSION="6.5.4"
wget -q https://lhapdf.hepforge.org/downloads/?f=LHAPDF-${LHAPDF_VERSION}.tar.gz \
    -O LHAPDF-${LHAPDF_VERSION}.tar.gz
tar xzf LHAPDF-${LHAPDF_VERSION}.tar.gz
cd LHAPDF-${LHAPDF_VERSION}
./configure --prefix=${HEP_DIR}/lhapdf
make -j$(nproc)
make install
cd ${HEP_DIR}

# Télécharger le set PDF nécessaire
${HEP_DIR}/lhapdf/bin/lhapdf install NNPDF31_nnlo_as_0118

export PATH="${HEP_DIR}/lhapdf/bin:$PATH"
export LD_LIBRARY_PATH="${HEP_DIR}/lhapdf/lib:${LD_LIBRARY_PATH:-}"
export LHAPDF_DATA_PATH="${HEP_DIR}/lhapdf/share/LHAPDF"

echo "  LHAPDF installé: $(lhapdf --version)"


# ========================= 3. MADGRAPH5_aMC@NLO =========================
echo ">>> Étape 3: MadGraph5_aMC@NLO"

# C'est un code Python — pas besoin de compiler le core!
MG5_VERSION="3.5.6"
wget -q https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v${MG5_VERSION}.tar.gz \
    -O MG5_aMC.tar.gz
tar xzf MG5_aMC.tar.gz
rm MG5_aMC.tar.gz

# Le dossier s'appelle souvent MG5_aMC_v3_5_6 ou similaire
MG5_DIR=$(ls -d ${HEP_DIR}/MG5_aMC_v* | head -1)

# Installer Pythia8 et Delphes directement DEPUIS MadGraph5
# (c'est la méthode recommandée et la plus simple!)
echo "  Installation de Pythia8 via MG5..."
echo "install pythia8" | python3 ${MG5_DIR}/bin/mg5_aMC

echo "  Installation de Delphes via MG5..."
echo "install Delphes" | python3 ${MG5_DIR}/bin/mg5_aMC

# Installer aussi le modèle loop_sm pour gg→HH
echo "  Installation du modèle loop_sm..."
echo "install loop_sm-no_b_mass" | python3 ${MG5_DIR}/bin/mg5_aMC

echo "  MadGraph5 installé dans: ${MG5_DIR}"


# ========================= 4. VÉRIFICATION =========================
echo ""
echo ">>> Vérification de l'installation"
echo ""

# Test MadGraph5
echo "  MadGraph5:"
python3 ${MG5_DIR}/bin/mg5_aMC --version 2>/dev/null || echo "    → Vérifier manuellement"

# Test Pythia8 (installé via MG5)
PYTHIA8_DIR="${MG5_DIR}/HEPTools/pythia8"
echo "  Pythia8: $(ls ${PYTHIA8_DIR}/bin/ 2>/dev/null | head -1 || echo 'vérifier dans HEPTools')"

# Test Delphes (installé via MG5)
DELPHES_DIR="${MG5_DIR}/Delphes"
echo "  Delphes: $(ls ${DELPHES_DIR}/DelphesHepMC2 2>/dev/null && echo 'OK' || echo 'vérifier')"

# Test ROOT
echo "  ROOT: $(root-config --version 2>/dev/null || echo 'non trouvé')"

# Test LHAPDF
echo "  LHAPDF: $(lhapdf --version 2>/dev/null || echo 'non trouvé')"


# ========================= 5. PYTHON PACKAGES =========================
echo ""
echo ">>> Étape 5: Packages Python pour l'analyse"

pip3 install --user \
    torch \
    uproot \
    awkward \
    h5py \
    scipy \
    scikit-learn \
    matplotlib \
    numpy \
    vector

echo "  Packages Python installés."


# ========================= 6. VARIABLES D'ENVIRONNEMENT =========================
echo ""
echo ">>> Étape 6: Configuration de l'environnement"

ENV_SCRIPT="${HEP_DIR}/setup_env.sh"
cat > ${ENV_SCRIPT} << ENVEOF
#!/bin/bash
# Source ce fichier avant d'utiliser la pipeline:
#   source ~/hep_tools/setup_env.sh

export HEP_DIR="${HEP_DIR}"
export MG5_DIR="${MG5_DIR}"
export PYTHIA8_DIR="${MG5_DIR}/HEPTools/pythia8"
export DELPHES_DIR="${MG5_DIR}/Delphes"
export ROOTSYS="${HEP_DIR}/root"

# ROOT
source \${ROOTSYS}/bin/thisroot.sh

# LHAPDF
export PATH="${HEP_DIR}/lhapdf/bin:\${PATH}"
export LD_LIBRARY_PATH="${HEP_DIR}/lhapdf/lib:\${LD_LIBRARY_PATH:-}"
export LHAPDF_DATA_PATH="${HEP_DIR}/lhapdf/share/LHAPDF"

# Pythia8
export PYTHIA8=\${PYTHIA8_DIR}
export LD_LIBRARY_PATH="\${PYTHIA8_DIR}/lib:\${LD_LIBRARY_PATH:-}"

echo "HEP environment loaded."
echo "  MG5:     \${MG5_DIR}"
echo "  Pythia8: \${PYTHIA8_DIR}"
echo "  Delphes: \${DELPHES_DIR}"
echo "  ROOT:    \$(root-config --version)"
ENVEOF

chmod +x ${ENV_SCRIPT}

echo ""
echo "============================================================"
echo "  INSTALLATION TERMINÉE"
echo ""
echo "  Avant chaque session, exécuter:"
echo "    source ~/hep_tools/setup_env.sh"
echo ""
echo "  Test rapide MadGraph5:"
echo "    echo 'generate p p > t t~' | python3 ${MG5_DIR}/bin/mg5_aMC"
echo "============================================================"
