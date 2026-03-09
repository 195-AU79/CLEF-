# Guide de Démarrage Rapide — CLEF Pipeline

## Prérequis

Installer Docker Desktop :
- **Windows** : https://docs.docker.com/desktop/install/windows-install/
  - Activer WSL2 quand demandé
- **macOS** : https://docs.docker.com/desktop/install/mac-install/
- **Linux** : https://docs.docker.com/engine/install/ubuntu/

Vérifier que Docker fonctionne :
```bash
docker --version
docker run hello-world
```

## Installation (une seule fois)

```bash
# 1. Se placer dans le dossier de la pipeline
cd clef_pipeline

# 2. Construire l'image (~30-45 min la première fois)
docker build -t clef-pipeline .

# 3. C'est tout !
```

L'image fait environ 8-10 Go et contient ROOT, MadGraph5, Pythia8, Delphes,
LHAPDF, PyTorch, et tous les outils Python nécessaires.

## Utilisation

### Mode interactif (recommandé pour commencer)
```bash
docker run -it --rm -v $(pwd)/results:/work/results clef-pipeline
```

Tu arrives dans un shell avec tout pré-configuré. Ensuite :

```bash
# Vérifier que tout marche
root-config --version
python3 -c "import torch; print(torch.__version__)"
ls $DELPHES_DIR/DelphesHepMC2

# Lancer un test MadGraph5
cd /work/clef_pipeline
echo "generate p p > t t~" | python3 $MG5_DIR/bin/mg5_aMC

# Lancer la pipeline complète
bash scripts/run_pipeline.sh all

# Ou étape par étape
bash scripts/run_pipeline.sh madgraph    # ~20 min
bash scripts/run_pipeline.sh shower      # ~10 min
bash scripts/run_pipeline.sh delphes     # ~5 min
bash scripts/run_pipeline.sh convert     # ~2 min
bash scripts/run_pipeline.sh train       # ~30 min (CPU), ~5 min (GPU)
bash scripts/run_pipeline.sh extract     # ~1 min
```

### Mode Docker Compose
```bash
# Démarrer
docker compose up -d

# Ouvrir un shell
docker compose exec clef bash

# Arrêter
docker compose down
```

### Avec GPU (pour l'entraînement ParT)
```bash
# Prérequis : NVIDIA Container Toolkit installé
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

docker run --gpus all -it --rm -v $(pwd)/results:/work/results clef-pipeline
```

## Récupérer les résultats

Les résultats sont montés dans `./results/` sur ta machine :
- `kl_extraction_results.npz` — données brutes du scan
- `kl_extraction_plots.png` — graphes (σ vs κ_λ, profile likelihood, m_HH)
- `best_model.pt` — modèle ParT entraîné

## Commandes utiles

```bash
# Voir les logs MG5
cat /work/clef_pipeline/data/signal/pp_HH_NLO/Events/run_01/run_01_tag_1_banner.txt

# Analyser le ROOT file Delphes interactivement
python3 -c "
import uproot
f = uproot.open('data/signal/pp_HH_NLO_delphes.root')
tree = f['Delphes']
print('Branches:', tree.keys())
print('N events:', tree.num_entries)
"

# Relancer juste l'extraction κ_λ avec une luminosité différente
python3 analysis/extract_kappa_lambda.py --luminosity 6000 --kl-true 1.5738738045 --plot

# Jupyter notebook (optionnel)
pip install jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

## Dépannage

**"Permission denied" sur les scripts :**
```bash
chmod +x scripts/*.sh
```

**MG5 ne trouve pas LHAPDF :**
```bash
echo "lhapdf = /opt/hep/lhapdf/bin/lhapdf-config" >> $MG5_DIR/input/mg5_configuration.txt
```

**Pas assez de mémoire :**
Réduire N_EVENTS dans run_pipeline.sh (mettre 10000 pour tester).

**Erreur de compilation Delphes/Pythia8 :**
Relancer depuis MG5 :
```bash
echo "install pythia8" | python3 $MG5_DIR/bin/mg5_aMC
echo "install Delphes" | python3 $MG5_DIR/bin/mg5_aMC
```
