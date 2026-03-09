########################################################################
#  Delphes Card — CLEF Pipeline
#  CMS-like detector for HL-LHC (14 TeV, 3000 fb⁻¹)
#  Optimized for HH → bbbb channel
########################################################################

#######################################
# Order of execution of various modules
#######################################

set ExecutionPath {
  ParticlePropagator
  ChargedHadronTrackingEfficiency
  ElectronTrackingEfficiency
  MuonTrackingEfficiency
  ChargedHadronMomentumSmearing
  ElectronEnergySmearing
  MuonMomentumSmearing
  TrackMerger
  Calorimeter
  EFlowMerger
  PhotonEfficiency
  PhotonIsolation
  ElectronEfficiency
  ElectronIsolation
  MuonEfficiency
  MuonIsolation
  MissingET
  GenJetFinder
  FastJetFinder
  FatJetFinder
  JetEnergyScale
  JetFlavorAssociation
  BTagging
  BTaggingLoose
  TauTagging
  ScalarHT
  TreeWriter
}

###############
# Propagator
###############
module ParticlePropagator ParticlePropagator {
  set InputArray Delphes/stableParticles
  set OutputArray stableParticles
  set ChargedHadronOutputArray chargedHadrons
  set ElectronOutputArray electrons
  set MuonOutputArray muons
  set Radius 1.29
  set HalfLength 3.0
  set Bz 3.8
}

#############################
# Charged Hadron Tracking
#############################
module Efficiency ChargedHadronTrackingEfficiency {
  set InputArray ParticlePropagator/chargedHadrons
  set OutputArray chargedHadrons
  set EfficiencyFormula {
    (pt <= 0.1)                                    * (0.00) +
    (abs(eta) <= 1.5) * (pt > 0.1 && pt <= 1.0)   * (0.70) +
    (abs(eta) <= 1.5) * (pt > 1.0 && pt <= 1.0e4) * (0.95) +
    (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1 && pt <= 1.0)   * (0.60) +
    (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 1.0 && pt <= 1.0e4) * (0.85) +
    (abs(eta) > 2.5)                               * (0.00)
  }
}

#############################
# Electron Tracking
#############################
module Efficiency ElectronTrackingEfficiency {
  set InputArray ParticlePropagator/electrons
  set OutputArray electrons
  set EfficiencyFormula {
    (pt <= 0.1)                                    * (0.00) +
    (abs(eta) <= 1.5) * (pt > 0.1 && pt <= 1.0)   * (0.73) +
    (abs(eta) <= 1.5) * (pt > 1.0 && pt <= 1.0e4) * (0.95) +
    (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1 && pt <= 1.0)   * (0.50) +
    (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 1.0 && pt <= 1.0e4) * (0.83) +
    (abs(eta) > 2.5)                               * (0.00)
  }
}

#############################
# Muon Tracking
#############################
module Efficiency MuonTrackingEfficiency {
  set InputArray ParticlePropagator/muons
  set OutputArray muons
  set EfficiencyFormula {
    (pt <= 0.1)                                    * (0.00) +
    (abs(eta) <= 1.5) * (pt > 0.1 && pt <= 1.0)   * (0.75) +
    (abs(eta) <= 1.5) * (pt > 1.0 && pt <= 1.0e4) * (0.99) +
    (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1 && pt <= 1.0)   * (0.70) +
    (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 1.0 && pt <= 1.0e4) * (0.98) +
    (abs(eta) > 2.5)                               * (0.00)
  }
}

############################
# Momentum Smearing
############################
module MomentumSmearing ChargedHadronMomentumSmearing {
  set InputArray ChargedHadronTrackingEfficiency/chargedHadrons
  set OutputArray chargedHadrons
  set ResolutionFormula {
    (abs(eta) <= 0.5) * (pt > 0.1) * sqrt(0.06^2 + pt^2*1.3e-3^2) +
    (abs(eta) > 0.5 && abs(eta) <= 1.5) * (pt > 0.1) * sqrt(0.10^2 + pt^2*2.0e-3^2) +
    (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1) * sqrt(0.25^2 + pt^2*5.0e-3^2)
  }
}

module EnergySmearing ElectronEnergySmearing {
  set InputArray ElectronTrackingEfficiency/electrons
  set OutputArray electrons
  set ResolutionFormula {
    (abs(eta) <= 2.5) * sqrt(energy^2*0.007^2 + energy*0.07^2 + 0.35^2)
  }
}

module MomentumSmearing MuonMomentumSmearing {
  set InputArray MuonTrackingEfficiency/muons
  set OutputArray muons
  set ResolutionFormula {
    (abs(eta) <= 0.5) * (pt > 0.1) * sqrt(0.01^2 + pt^2*1.0e-4^2) +
    (abs(eta) > 0.5 && abs(eta) <= 1.5) * (pt > 0.1) * sqrt(0.02^2 + pt^2*1.5e-4^2) +
    (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1) * sqrt(0.05^2 + pt^2*2.5e-4^2)
  }
}

######################
# Track Merger
######################
module Merger TrackMerger {
  add InputArray ChargedHadronMomentumSmearing/chargedHadrons
  add InputArray ElectronEnergySmearing/electrons
  add InputArray MuonMomentumSmearing/muons
  set OutputArray tracks
}

######################
# Calorimeter
######################
module Calorimeter Calorimeter {
  set ParticleInputArray ParticlePropagator/stableParticles
  set TrackInputArray TrackMerger/tracks
  set TowerOutputArray towers
  set PhotonOutputArray photons
  set EFlowTrackOutputArray eflowTracks
  set EFlowPhotonOutputArray eflowPhotons
  set EFlowNeutralHadronOutputArray eflowNeutralHadrons
  set ECalEnergyMin 0.5
  set HCalEnergyMin 1.0
  set ECalEnergySignificanceMin 1.0
  set HCalEnergySignificanceMin 1.0
  set SmearTowerCenter true

  # ECAL eta,phi segmentation
  set EtaPhiBins {
    (-4.0, -3.0, 80) (-3.0, -2.5, 40) (-2.5, -1.5, 40)
    (-1.5, -1.0, 40) (-1.0, 0.0, 40) (0.0, 1.0, 40)
    (1.0, 1.5, 40) (1.5, 2.5, 40) (2.5, 3.0, 40) (3.0, 4.0, 80)
  }

  # ECAL resolution
  set ECalResolutionFormula {
    (abs(eta) <= 3.0) * sqrt(energy^2*0.007^2 + energy*0.07^2 + 0.35^2) +
    (abs(eta) > 3.0 && abs(eta) <= 5.0) * sqrt(energy^2*0.107^2 + energy*2.08^2)
  }

  # HCAL resolution
  set HCalResolutionFormula {
    (abs(eta) <= 3.0) * sqrt(energy^2*0.050^2 + energy*1.50^2) +
    (abs(eta) > 3.0 && abs(eta) <= 5.0) * sqrt(energy^2*0.090^2 + energy*2.13^2)
  }
}

######################
# EFlow Merger
######################
module Merger EFlowMerger {
  add InputArray Calorimeter/eflowTracks
  add InputArray Calorimeter/eflowPhotons
  add InputArray Calorimeter/eflowNeutralHadrons
  set OutputArray eflow
}

######################
# Photon / Electron / Muon efficiency + isolation
######################
module Efficiency PhotonEfficiency { set InputArray Calorimeter/photons; set OutputArray photons; set EfficiencyFormula { (pt > 10 && abs(eta) < 2.5) * 0.95 } }
module Isolation PhotonIsolation { set CandidateInputArray PhotonEfficiency/photons; set IsolationInputArray EFlowMerger/eflow; set OutputArray photons; set DeltaRMax 0.3; set PTMin 0.5; set PTRatioMax 0.12 }
module Efficiency ElectronEfficiency { set InputArray ElectronEnergySmearing/electrons; set OutputArray electrons; set EfficiencyFormula { (pt > 10 && abs(eta) < 2.5) * 0.85 } }
module Isolation ElectronIsolation { set CandidateInputArray ElectronEfficiency/electrons; set IsolationInputArray EFlowMerger/eflow; set OutputArray electrons; set DeltaRMax 0.3; set PTMin 0.5; set PTRatioMax 0.12 }
module Efficiency MuonEfficiency { set InputArray MuonMomentumSmearing/muons; set OutputArray muons; set EfficiencyFormula { (pt > 10 && abs(eta) < 2.4) * 0.95 } }
module Isolation MuonIsolation { set CandidateInputArray MuonEfficiency/muons; set IsolationInputArray EFlowMerger/eflow; set OutputArray muons; set DeltaRMax 0.3; set PTMin 0.5; set PTRatioMax 0.25 }

######################
# Missing ET
######################
module Merger MissingET {
  add InputArray EFlowMerger/eflow
  set MomentumOutputArray momentum
}

#################
# Gen Jet Finder (anti-kT R=0.4)
#################
module FastJetFinder GenJetFinder {
  set InputArray Delphes/stableParticles
  set OutputArray jets
  set JetAlgorithm 6
  set ParameterR 0.4
  set JetPTMin 20.0
}

#################
# Jet Finder (anti-kT R=0.4) — main jets for bbbb
#################
module FastJetFinder FastJetFinder {
  set InputArray EFlowMerger/eflow
  set OutputArray jets
  set JetAlgorithm 6        ! anti-kT
  set ParameterR 0.4
  set JetPTMin 20.0
  set ConeRadius 0.4
  set SeedThreshold 1.0
}

#################
# Fat Jet Finder (anti-kT R=1.0) — for boosted HH
#################
module FastJetFinder FatJetFinder {
  set InputArray EFlowMerger/eflow
  set OutputArray jets
  set JetAlgorithm 6        ! anti-kT
  set ParameterR 1.0
  set JetPTMin 200.0
  set ComputeNsubjettiness 1
  set Beta 1.0
  set AxisMode 4
  set ComputeSoftDrop 1
  set BetaSoftDrop 0.0
  set SymmetryCutSoftDrop 0.1
  set R0SoftDrop 1.0
}

######################
# Jet Energy Scale
######################
module EnergyScale JetEnergyScale {
  set InputArray FastJetFinder/jets
  set OutputArray jets
  set ScaleFormula {
    1.0  ! Nominal (systematics via NP)
  }
}

######################
# Jet Flavor Association
######################
module JetFlavorAssociation JetFlavorAssociation {
  set PartonInputArray Delphes/partons
  set ParticleInputArray Delphes/allParticles
  set ParticleLHEFInputArray Delphes/allParticlesLHEF
  set JetInputArray JetEnergyScale/jets
  set DeltaR 0.5
  set PartonPTMin 1.0
  set PartonEtaMax 2.5
}

######################
# B-Tagging — TIGHT (DeepJet WP: εb=70%, misstag~0.1%)
# Critical for HH → bbbb
######################
module BTagging BTagging {
  set JetInputArray JetFlavorAssociation/jets
  set BitNumber 0

  # CMS Phase-2 DeepJet-like tight WP
  add EfficiencyFormula {0} {
    (pt > 20 && pt <= 50)   * (abs(eta) <= 2.5) * 0.001 +
    (pt > 50 && pt <= 100)  * (abs(eta) <= 2.5) * 0.001 +
    (pt > 100 && pt <= 200) * (abs(eta) <= 2.5) * 0.001 +
    (pt > 200)              * (abs(eta) <= 2.5) * 0.002
  }

  # c-jet mistag
  add EfficiencyFormula {4} {
    (pt > 20 && pt <= 50)   * (abs(eta) <= 2.5) * 0.04 +
    (pt > 50 && pt <= 100)  * (abs(eta) <= 2.5) * 0.03 +
    (pt > 100 && pt <= 200) * (abs(eta) <= 2.5) * 0.02 +
    (pt > 200)              * (abs(eta) <= 2.5) * 0.02
  }

  # b-jet efficiency (tight)
  add EfficiencyFormula {5} {
    (pt > 20 && pt <= 50)   * (abs(eta) <= 2.5) * 0.65 +
    (pt > 50 && pt <= 100)  * (abs(eta) <= 2.5) * 0.75 +
    (pt > 100 && pt <= 200) * (abs(eta) <= 2.5) * 0.70 +
    (pt > 200)              * (abs(eta) <= 2.5) * 0.60
  }
}

######################
# B-Tagging — LOOSE (for veto / 4b selection)
######################
module BTagging BTaggingLoose {
  set JetInputArray JetFlavorAssociation/jets
  set BitNumber 1

  add EfficiencyFormula {0} {
    (pt > 20 && abs(eta) <= 2.5) * 0.01
  }

  add EfficiencyFormula {4} {
    (pt > 20 && abs(eta) <= 2.5) * 0.15
  }

  add EfficiencyFormula {5} {
    (pt > 20 && abs(eta) <= 2.5) * 0.90
  }
}

######################
# Tau Tagging
######################
module TauTagging TauTagging {
  set ParticleInputArray Delphes/allParticles
  set PartonInputArray Delphes/partons
  set JetInputArray JetEnergyScale/jets
  set DeltaR 0.5
  set TauPTMin 20.0
  set TauEtaMax 2.3
  add EfficiencyFormula {0} { 0.001 }
  add EfficiencyFormula {15} {
    (pt > 20 && abs(eta) < 2.3) * 0.60
  }
}

######################
# Scalar HT
######################
module Merger ScalarHT {
  add InputArray JetEnergyScale/jets
  add InputArray ElectronIsolation/electrons
  add InputArray MuonIsolation/muons
  set EnergyOutputArray energy
}

######################
# Tree Writer — OUTPUT
######################
module TreeWriter TreeWriter {
  # Particles
  add Branch Delphes/allParticles        Particle     GenParticle
  
  # Jets (main output for bbbb)
  add Branch JetEnergyScale/jets         Jet          Jet
  add Branch FatJetFinder/jets           FatJet       Jet
  add Branch GenJetFinder/jets           GenJet       Jet
  
  # Leptons (for veto)
  add Branch ElectronIsolation/electrons Electron     Electron
  add Branch MuonIsolation/muons         Muon         Muon
  add Branch PhotonIsolation/photons     Photon       Photon
  
  # MET
  add Branch MissingET/momentum          MissingET    MissingET
  
  # HT
  add Branch ScalarHT/energy             ScalarHT     ScalarHT
}
