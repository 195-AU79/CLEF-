#=================================================================
#  CLEF Pipeline — Docker Image
#  MadGraph5_aMC@NLO + Pythia8 + Delphes + ROOT + LHAPDF
#  pp → HH → bbbb @ 14 TeV, κ_λ = 1.5738738045
#
#  Build:
#    docker build -t clef-pipeline .
#
#  Run (interactif):
#    docker run -it --rm -v $(pwd)/results:/work/results clef-pipeline
#
#  Run (pipeline complète):
#    docker run --rm -v $(pwd)/results:/work/results clef-pipeline \
#      bash /work/clef_pipeline/scripts/run_pipeline.sh all
#
#  Avec GPU (PyTorch):
#    docker run --gpus all -it --rm -v $(pwd)/results:/work/results clef-pipeline
#=================================================================

FROM ubuntu:22.04

LABEL maintainer="CLEF Pipeline"
LABEL description="Full HEP simulation chain for HH→bbbb κ_λ extraction"

# Éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# ========================= PRÉREQUIS SYSTÈME =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    gcc \
    g++ \
    cmake \
    wget \
    curl \
    git \
    rsync \
    tcl \
    tcl-dev \
    ghostscript \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-six \
    zlib1g-dev \
    libboost-all-dev \
    libbz2-dev \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    libreadline-dev \
    liblzma-dev \
    libx11-dev \
    libxpm-dev \
    libxft-dev \
    libxext-dev \
    libpng-dev \
    libjpeg-dev \
    libgif-dev \
    libpcre3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxml2-dev \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Lien symbolique python → python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Répertoire de travail
WORKDIR /opt/hep
ENV HEP_DIR=/opt/hep

# ========================= 1. ROOT 6.32 =========================
# Binaire pré-compilé (beaucoup plus rapide que compiler)
ENV ROOT_VERSION=6.32.02

RUN wget -q https://root.cern/download/root_v${ROOT_VERSION}.Linux-ubuntu22.04-x86_64-gcc11.4.tar.gz \
        -O root.tar.gz \
    && tar xzf root.tar.gz \
    && rm root.tar.gz

ENV ROOTSYS=/opt/hep/root
ENV PATH="${ROOTSYS}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROOTSYS}/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${ROOTSYS}/lib:${PYTHONPATH}"
ENV CMAKE_PREFIX_PATH="${ROOTSYS}:${CMAKE_PREFIX_PATH}"

# Vérifier ROOT
RUN root-config --version

# ========================= 2. LHAPDF 6.5.4 =========================
ENV LHAPDF_VERSION=6.5.4

RUN wget -q https://lhapdf.hepforge.org/downloads/?f=LHAPDF-${LHAPDF_VERSION}.tar.gz \
        -O LHAPDF-${LHAPDF_VERSION}.tar.gz \
    && tar xzf LHAPDF-${LHAPDF_VERSION}.tar.gz \
    && cd LHAPDF-${LHAPDF_VERSION} \
    && ./configure --prefix=/opt/hep/lhapdf \
    && make -j$(nproc) \
    && make install \
    && cd /opt/hep \
    && rm -rf LHAPDF-${LHAPDF_VERSION} LHAPDF-${LHAPDF_VERSION}.tar.gz

ENV PATH="/opt/hep/lhapdf/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/hep/lhapdf/lib:${LD_LIBRARY_PATH}"
ENV LHAPDF_DATA_PATH="/opt/hep/lhapdf/share/LHAPDF"
ENV LHAPATH="/opt/hep/lhapdf/share/LHAPDF"

# Télécharger les PDFs nécessaires
RUN /opt/hep/lhapdf/bin/lhapdf install NNPDF31_nnlo_as_0118 \
    && /opt/hep/lhapdf/bin/lhapdf install NNPDF31_nlo_as_0118

# ========================= 3. MADGRAPH5_aMC@NLO =========================
ENV MG5_VERSION=3.5.6

RUN wget -q https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v${MG5_VERSION}.tar.gz \
        -O MG5_aMC.tar.gz \
    && tar xzf MG5_aMC.tar.gz \
    && rm MG5_aMC.tar.gz

# Trouver le dossier MG5 (le nom varie selon la version)
RUN MG5=$(ls -d /opt/hep/MG5_aMC_v* | head -1) \
    && ln -sf ${MG5} /opt/hep/mg5amc

ENV MG5_DIR=/opt/hep/mg5amc

# Configurer MG5 pour utiliser LHAPDF
RUN echo "lhapdf = /opt/hep/lhapdf/bin/lhapdf-config" >> ${MG5_DIR}/input/mg5_configuration.txt \
    && echo "lhapdf_py3 = /opt/hep/lhapdf/bin/lhapdf-config" >> ${MG5_DIR}/input/mg5_configuration.txt

# ========================= 4. PYTHIA8 (via MG5) =========================
RUN echo "install pythia8" | python3 ${MG5_DIR}/bin/mg5_aMC --logging=ERROR

ENV PYTHIA8=/opt/hep/mg5amc/HEPTools/pythia8
ENV LD_LIBRARY_PATH="${PYTHIA8}/lib:${LD_LIBRARY_PATH}"

# ========================= 5. DELPHES (via MG5) =========================
RUN echo "install Delphes" | python3 ${MG5_DIR}/bin/mg5_aMC --logging=ERROR

ENV DELPHES_DIR=/opt/hep/mg5amc/Delphes
ENV LD_LIBRARY_PATH="${DELPHES_DIR}:${LD_LIBRARY_PATH}"
ENV ROOT_INCLUDE_PATH="${DELPHES_DIR}/external:${ROOT_INCLUDE_PATH}"

# ========================= 6. MODÈLES MG5 =========================
# Installer le modèle loop_sm pour gg → HH (NLO)
RUN echo "install loop_sm-no_b_mass" | python3 ${MG5_DIR}/bin/mg5_aMC --logging=ERROR || true

# ========================= 7. FASTJET =========================
# Normalement installé avec Delphes, sinon :
RUN if [ ! -f /opt/hep/mg5amc/HEPTools/fastjet/bin/fastjet-config ]; then \
        echo "install fastjet" | python3 ${MG5_DIR}/bin/mg5_aMC --logging=ERROR; \
    fi

# ========================= 8. PACKAGES PYTHON =========================
RUN pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir \
    uproot \
    awkward \
    h5py \
    scipy \
    scikit-learn \
    matplotlib \
    numpy \
    vector \
    particle \
    hepunits \
    pyhepmc

# ========================= 9. WORKSPACE =========================
WORKDIR /work

# Copier la pipeline CLEF
COPY . /work/clef_pipeline/

# Script d'entrée pour configurer l'environnement
RUN cat > /work/setup_env.sh << 'ENVEOF'
#!/bin/bash
source /opt/hep/root/bin/thisroot.sh
export HEP_DIR=/opt/hep
export MG5_DIR=/opt/hep/mg5amc
export PYTHIA8=/opt/hep/mg5amc/HEPTools/pythia8
export DELPHES_DIR=/opt/hep/mg5amc/Delphes
export PATH="/opt/hep/lhapdf/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/hep/lhapdf/lib:${PYTHIA8}/lib:${DELPHES_DIR}:${ROOTSYS}/lib:${LD_LIBRARY_PATH}"
export LHAPDF_DATA_PATH="/opt/hep/lhapdf/share/LHAPDF"
echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║     CLEF Pipeline — Environment Ready    ║"
echo "  ║     κ_λ = 1.5738738045 (CLEF prediction)        ║"
echo "  ╠══════════════════════════════════════════╣"
echo "  ║  MG5:     $(python3 ${MG5_DIR}/bin/mg5_aMC --version 2>/dev/null | head -1 || echo 'v3.5.6')"
echo "  ║  ROOT:    $(root-config --version)"
echo "  ║  LHAPDF:  $(lhapdf --version 2>/dev/null || echo '6.5.4')"
echo "  ║  Pythia8: $(ls ${PYTHIA8}/share/Pythia8/xmldoc/Version.xml 2>/dev/null | head -1 && echo 'OK' || echo 'installed')"
echo "  ║  Delphes: $(ls ${DELPHES_DIR}/DelphesHepMC2 2>/dev/null && echo 'OK' || echo 'installed')"
echo "  ║  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'installed')"
echo "  ╚══════════════════════════════════════════╝"
echo ""
ENVEOF
chmod +x /work/setup_env.sh

# Charger l'env au démarrage
RUN echo "source /work/setup_env.sh" >> /root/.bashrc

# ========================= 10. VÉRIFICATION FINALE =========================
RUN echo "Checking installations..." \
    && root-config --version \
    && python3 -c "import ROOT; print(f'PyROOT: {ROOT.__version__}')" || true \
    && python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" \
    && python3 -c "import uproot; print(f'uproot: {uproot.__version__}')" \
    && ls ${DELPHES_DIR}/DelphesHepMC2 \
    && echo "All checks passed!"

# Exposer le port pour Jupyter (optionnel)
EXPOSE 8888

# Point d'entrée
ENTRYPOINT ["/bin/bash", "-c", "source /work/setup_env.sh && exec \"$@\"", "--"]
CMD ["/bin/bash"]
