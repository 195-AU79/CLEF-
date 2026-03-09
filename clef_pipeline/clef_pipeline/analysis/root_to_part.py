#!/usr/bin/env python3
"""
CLEF Pipeline — Step 4: ROOT → ParT Feature Extraction
Convert Delphes ROOT output to numpy arrays for Particle Transformer training.

Features per event:
  - Jet-level: pT, eta, phi, mass, btag, N constituents
  - Constituent-level (per jet): pT, eta, phi, pid, d0, dz
  - Event-level: HT, MET, n_jets, n_bjets, mHH, m_H1, m_H2,
                 deltaR_H1, deltaR_H2, deltaEta_HH, chi2_pairing

Output: .h5 files with jet constituents + event features + reweight info

Usage:
    python root_to_part.py --input delphes_output.root --output features.h5
"""

import argparse
import numpy as np
import sys
from pathlib import Path

try:
    import uproot
    import awkward as ak
except ImportError:
    print("ERROR: install uproot + awkward: pip install uproot awkward")
    sys.exit(1)

try:
    import h5py
except ImportError:
    print("ERROR: install h5py: pip install h5py")
    sys.exit(1)


# ============================================================
# Constants
# ============================================================
M_HIGGS = 125.09  # GeV
N_CONSTITUENTS_MAX = 50   # Max constituents per jet
N_JETS_MAX = 6            # Max jets per event (≥4 for bbbb)
FEATURE_DIM = 7           # pT, eta, phi, pid, d0, dz, btag_score


def load_delphes_root(filepath):
    """Load Delphes ROOT file with uproot."""
    f = uproot.open(filepath)
    tree = f["Delphes"]
    return tree


def extract_jet_features(tree, n_events=None):
    """
    Extract jet-level and constituent-level features from Delphes tree.
    
    Returns:
        jet_features: (N_events, N_jets_max, N_jet_features)
        constituents: (N_events, N_jets_max, N_constituents_max, Feature_dim)
        event_features: (N_events, N_event_features)
        labels: (N_events,) — signal/background label
        weights: (N_events, N_kl_points) — reweighting factors
    """
    # Read jet arrays
    jet_pt = tree["Jet.PT"].array(entry_stop=n_events)
    jet_eta = tree["Jet.Eta"].array(entry_stop=n_events)
    jet_phi = tree["Jet.Phi"].array(entry_stop=n_events)
    jet_mass = tree["Jet.Mass"].array(entry_stop=n_events)
    jet_btag = tree["Jet.BTag"].array(entry_stop=n_events)
    
    # MET
    met = tree["MissingET.MET"].array(entry_stop=n_events)
    
    # Number of events
    n_evt = len(jet_pt)
    
    # Initialize output arrays
    jets = np.zeros((n_evt, N_JETS_MAX, 5), dtype=np.float32)  # pT, eta, phi, mass, btag
    event_feats = np.zeros((n_evt, 12), dtype=np.float32)
    mask = np.zeros((n_evt, N_JETS_MAX), dtype=bool)
    
    for i in range(n_evt):
        n_j = min(len(jet_pt[i]), N_JETS_MAX)
        
        # Sort by pT descending
        idx_sort = np.argsort(-np.array(jet_pt[i]))[:n_j]
        
        for j_idx, orig_idx in enumerate(idx_sort):
            jets[i, j_idx, 0] = jet_pt[i][orig_idx]
            jets[i, j_idx, 1] = jet_eta[i][orig_idx]
            jets[i, j_idx, 2] = jet_phi[i][orig_idx]
            jets[i, j_idx, 3] = jet_mass[i][orig_idx]
            jets[i, j_idx, 4] = jet_btag[i][orig_idx]
            mask[i, j_idx] = True
        
        # Event-level features
        n_bjets = int(np.sum(np.array(jet_btag[i]) > 0))
        ht = float(np.sum(np.array(jet_pt[i])))
        
        event_feats[i, 0] = ht
        event_feats[i, 1] = float(met[i][0]) if len(met[i]) > 0 else 0.0
        event_feats[i, 2] = n_j
        event_feats[i, 3] = n_bjets
        
        # Higgs candidate reconstruction (chi2 pairing)
        if n_j >= 4 and n_bjets >= 3:
            m_h1, m_h2, dr_h1, dr_h2, deta_hh, mhh, chi2 = reconstruct_higgs(
                jets[i, :n_j, :4]
            )
            event_feats[i, 4] = m_h1
            event_feats[i, 5] = m_h2
            event_feats[i, 6] = dr_h1
            event_feats[i, 7] = dr_h2
            event_feats[i, 8] = deta_hh
            event_feats[i, 9] = mhh
            event_feats[i, 10] = chi2
            event_feats[i, 11] = 1.0  # valid pairing flag
    
    return jets, event_feats, mask


def reconstruct_higgs(jets_4vec):
    """
    Reconstruct two Higgs candidates from 4 leading jets using
    chi-squared minimization of |m_jj - m_H|.
    
    Args:
        jets_4vec: (N_jets, 4) array of [pT, eta, phi, mass]
    
    Returns:
        m_H1, m_H2, deltaR_H1, deltaR_H2, deltaEta_HH, m_HH, chi2
    """
    from itertools import combinations
    
    def p4_from_ptEtaPhiM(pt, eta, phi, m):
        """Convert to (px, py, pz, E)."""
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(px**2 + py**2 + pz**2 + m**2)
        return np.array([px, py, pz, E])
    
    def inv_mass(p1, p2):
        p = p1 + p2
        return np.sqrt(max(0, p[3]**2 - p[0]**2 - p[1]**2 - p[2]**2))
    
    def delta_r(eta1, phi1, eta2, phi2):
        deta = eta1 - eta2
        dphi = phi1 - phi2
        # Wrap dphi
        while dphi > np.pi: dphi -= 2*np.pi
        while dphi < -np.pi: dphi += 2*np.pi
        return np.sqrt(deta**2 + dphi**2)
    
    # Take first 4 jets
    n = min(4, len(jets_4vec))
    if n < 4:
        return 0, 0, 0, 0, 0, 0, 999.0
    
    p4s = [p4_from_ptEtaPhiM(*jets_4vec[k]) for k in range(4)]
    
    # 3 possible pairings of 4 jets into 2 pairs
    pairings = [
        ((0,1), (2,3)),
        ((0,2), (1,3)),
        ((0,3), (1,2)),
    ]
    
    best_chi2 = 1e9
    best_pairing = None
    
    sigma_m = 15.0  # GeV — jet mass resolution
    
    for (i1, i2), (i3, i4) in pairings:
        m1 = inv_mass(p4s[i1], p4s[i2])
        m2 = inv_mass(p4s[i3], p4s[i4])
        chi2 = ((m1 - M_HIGGS) / sigma_m)**2 + ((m2 - M_HIGGS) / sigma_m)**2
        
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_pairing = ((i1, i2), (i3, i4))
            best_m1, best_m2 = m1, m2
    
    (i1, i2), (i3, i4) = best_pairing
    
    # Ensure m_H1 > m_H2 convention
    if best_m1 < best_m2:
        best_m1, best_m2 = best_m2, best_m1
        (i1, i2), (i3, i4) = (i3, i4), (i1, i2)
    
    dr_h1 = delta_r(jets_4vec[i1,1], jets_4vec[i1,2],
                     jets_4vec[i2,1], jets_4vec[i2,2])
    dr_h2 = delta_r(jets_4vec[i3,1], jets_4vec[i3,2],
                     jets_4vec[i4,1], jets_4vec[i4,2])
    
    # HH system
    p_hh = p4s[i1] + p4s[i2] + p4s[i3] + p4s[i4]
    m_hh = np.sqrt(max(0, p_hh[3]**2 - p_hh[0]**2 - p_hh[1]**2 - p_hh[2]**2))
    
    eta_h1 = (jets_4vec[i1,1] + jets_4vec[i2,1]) / 2.0  # approx
    eta_h2 = (jets_4vec[i3,1] + jets_4vec[i4,1]) / 2.0
    deta_hh = abs(eta_h1 - eta_h2)
    
    return best_m1, best_m2, dr_h1, dr_h2, deta_hh, m_hh, best_chi2


def save_h5(output_path, jets, event_feats, mask, label=1, kl_value=1.5738738045):
    """Save features to HDF5 for ParT training."""
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('jets', data=jets, compression='gzip')
        f.create_dataset('event_features', data=event_feats, compression='gzip')
        f.create_dataset('jet_mask', data=mask, compression='gzip')
        f.create_dataset('labels', data=np.full(len(jets), label, dtype=np.int32))
        
        # Store κ_λ value as attribute
        f.attrs['kappa_lambda'] = kl_value
        f.attrs['process'] = 'pp_HH_bbbb'
        f.attrs['sqrt_s'] = 14000.0
        f.attrs['n_events'] = len(jets)
        
        # Feature names for reference
        f.attrs['jet_features'] = ['pT', 'eta', 'phi', 'mass', 'btag']
        f.attrs['event_features'] = [
            'HT', 'MET', 'n_jets', 'n_bjets',
            'm_H1', 'm_H2', 'dR_H1', 'dR_H2',
            'dEta_HH', 'm_HH', 'chi2_pairing', 'valid_pairing'
        ]
    
    print(f"Saved {len(jets)} events to {output_path}")
    print(f"  κ_λ = {kl_value}")
    print(f"  Jets shape: {jets.shape}")
    print(f"  Event features shape: {event_feats.shape}")


def main():
    parser = argparse.ArgumentParser(description="CLEF Pipeline: ROOT → ParT features")
    parser.add_argument('--input', '-i', required=True, help='Input Delphes ROOT file')
    parser.add_argument('--output', '-o', required=True, help='Output HDF5 file')
    parser.add_argument('--label', type=int, default=1, help='Label: 1=signal, 0=background')
    parser.add_argument('--kl', type=float, default=1.5738738045, help='κ_λ value for this sample')
    parser.add_argument('--max-events', type=int, default=None, help='Max events to process')
    
    args = parser.parse_args()
    
    print(f"CLEF Pipeline — ROOT → ParT Feature Extraction")
    print(f"{'='*50}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"κ_λ:    {args.kl}")
    print()
    
    tree = load_delphes_root(args.input)
    jets, event_feats, mask = extract_jet_features(tree, args.max_events)
    save_h5(args.output, jets, event_feats, mask, args.label, args.kl)


if __name__ == '__main__':
    main()
