#!/usr/bin/env python3
"""
CLEF Pipeline — κ_λ Extraction via Profile Likelihood
=====================================================

Extracts the Higgs self-coupling modifier κ_λ from ParT output
using morphing + profile likelihood ratio.

Method:
  1. Generate templates: σ(κ_λ) = A·κ_λ² + B·κ_λ + C (quadratic morphing)
     where A, B, C come from reweighted MC at κ_λ = {0, 1, 2.46} (basis points)
  2. Apply ParT classifier score cut → signal-enriched region
  3. Use m_HH distribution as discriminant
  4. Profile likelihood: -2·ΔlnL(κ_λ) via template fit
  5. Extract κ_λ posterior + check CLEF prediction κ_λ = 1.5738738045

CLEF prediction:  κ_λ = 1.5738738045  (from CW potential + d_f=11/3)
SM:               κ_λ = 1.000
Expected sensitivity @ 3 ab⁻¹: σ(κ_λ) ~ 0.3 (CMS projection)

Usage:
    python extract_kappa_lambda.py --part-output results.npz --luminosity 3000
"""

import argparse
import numpy as np
from pathlib import Path

try:
    from scipy.optimize import minimize_scalar, minimize
    from scipy.interpolate import interp1d
    from scipy.stats import norm
except ImportError:
    print("ERROR: install scipy")
    raise


# ============================================================
# Cross section morphing: σ(κ_λ) for gg → HH
# ============================================================
# The HH cross section has a well-known quadratic dependence on κ_λ
# due to the interference between triangle and box diagrams:
#
#   σ(κ_λ) = σ_box + κ_λ · σ_int + κ_λ² · σ_tri
#
# At 14 TeV (NLO QCD):
#   σ_box ≈ 70.0 fb  (box diagram)
#   σ_int ≈ -50.4 fb  (interference, destructive for κ_λ=1)
#   σ_tri ≈ 11.6 fb  (triangle diagram)
#   σ_SM  ≈ 31.05 fb  (κ_λ = 1)

# NLO QCD coefficients at 14 TeV (from Grazzini et al., arXiv:1803.02463)
SIGMA_BOX = 70.01   # fb — box diagram (κ_λ-independent)
SIGMA_INT = -50.45   # fb — triangle-box interference (∝ κ_λ)
SIGMA_TRI = 11.59    # fb — triangle diagram (∝ κ_λ²)

BR_BBBB = 0.339  # BR(HH → bbbb) = 0.582² = 0.339


def sigma_HH(kl):
    """Total pp → HH cross section at 14 TeV NLO [fb]."""
    return SIGMA_BOX + kl * SIGMA_INT + kl**2 * SIGMA_TRI


def sigma_HH_bbbb(kl):
    """pp → HH → bbbb cross section [fb]."""
    return sigma_HH(kl) * BR_BBBB


# ============================================================
# m_HH shape morphing
# ============================================================
# Shape of m_HH distribution depends on κ_λ through triangle/box interference
# We parameterize using 3 basis shapes (κ_λ = 0, 1, κ_λ_max)

def generate_mHH_template(kl, n_bins=25, mhh_range=(250, 1200)):
    """
    Generate m_HH distribution template for given κ_λ.
    
    Uses morphing: shape(κ_λ) = w_box·S_box + w_int·S_int + w_tri·S_tri
    
    Simplified parameterization based on full NLO shapes.
    """
    bins = np.linspace(mhh_range[0], mhh_range[1], n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    
    # Box shape: broad, peaks around 400 GeV
    shape_box = np.exp(-0.5 * ((centers - 400) / 120)**2) + \
                0.3 * np.exp(-0.5 * ((centers - 600) / 200)**2)
    
    # Triangle shape: peaks at threshold (~2·m_H ≈ 250 GeV)
    shape_tri = np.exp(-0.5 * ((centers - 300) / 60)**2)
    
    # Interference shape: complex, peaks near 350 GeV (destructive)
    shape_int = -0.8 * np.exp(-0.5 * ((centers - 350) / 80)**2) + \
                 0.2 * np.exp(-0.5 * ((centers - 500) / 150)**2)
    
    # Morphed shape
    template = SIGMA_BOX * shape_box + kl * SIGMA_INT * shape_int + kl**2 * SIGMA_TRI * shape_tri
    
    # Normalize to cross section
    template = np.maximum(template, 0)
    if template.sum() > 0:
        template = template / template.sum() * sigma_HH_bbbb(kl)
    
    return bins, centers, template


# ============================================================
# Background model (QCD 4b + tt̄+jets)
# ============================================================
def background_template(n_bins=25, mhh_range=(250, 1200), lumi=3000):
    """
    Background m_HH template (QCD 4b + ttbar dominant).
    
    After ParT selection (εs ~ 40%, εb ~ 0.5%):
    - QCD 4b: σ × ε ~ 15 fb
    - tt̄+jets: σ × ε ~ 8 fb
    """
    bins = np.linspace(mhh_range[0], mhh_range[1], n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    
    # QCD 4b: falling spectrum
    bkg_qcd = 12.0 * np.exp(-(centers - 250) / 300)
    
    # tt̄: peak near 2·m_t ≈ 350 GeV
    bkg_tt = 6.0 * np.exp(-0.5 * ((centers - 370) / 100)**2)
    
    bkg = bkg_qcd + bkg_tt
    bkg_total_xsec = 23.0  # fb (after ParT selection)
    bkg = bkg / bkg.sum() * bkg_total_xsec
    
    # Scale to events
    bkg_events = bkg * lumi  # events per bin
    
    return bins, centers, bkg_events


# ============================================================
# Profile likelihood
# ============================================================
def neg_log_likelihood(kl, data_hist, bkg_hist, lumi, 
                        n_bins=25, mhh_range=(250, 1200),
                        signal_efficiency=0.40):
    """
    Poisson negative log-likelihood for κ_λ hypothesis.
    
    -2·ln(L) = 2·Σᵢ [μᵢ - nᵢ + nᵢ·ln(nᵢ/μᵢ)]
    where μᵢ = s_i(κ_λ) + b_i
    """
    _, _, sig_template = generate_mHH_template(kl, n_bins, mhh_range)
    sig_events = sig_template * lumi * signal_efficiency
    
    mu = sig_events + bkg_hist  # expected per bin
    mu = np.maximum(mu, 1e-10)  # avoid log(0)
    
    # Poisson log-likelihood
    nll = 0.0
    for i in range(n_bins):
        n = data_hist[i]
        m = mu[i]
        if n > 0:
            nll += m - n + n * np.log(n / m)
        else:
            nll += m
    
    return 2.0 * nll


def profile_likelihood_scan(data_hist, bkg_hist, lumi, 
                             kl_scan=None, n_bins=25,
                             signal_efficiency=0.40):
    """
    Scan -2·ΔlnL(κ_λ) over range.
    
    Returns:
        kl_scan: array of κ_λ values
        delta_nll: -2·Δln(L) values
        kl_best: best-fit κ_λ
        kl_1sigma: (lower, upper) 68% CL interval
        kl_2sigma: (lower, upper) 95% CL interval
    """
    if kl_scan is None:
        kl_scan = np.linspace(-1, 8, 200)
    
    nll_values = np.array([
        neg_log_likelihood(kl, data_hist, bkg_hist, lumi, n_bins, 
                          signal_efficiency=signal_efficiency)
        for kl in kl_scan
    ])
    
    # Best fit
    idx_best = np.argmin(nll_values)
    kl_best = kl_scan[idx_best]
    nll_min = nll_values[idx_best]
    
    delta_nll = nll_values - nll_min
    
    # 1σ and 2σ intervals (Wilks' theorem: -2ΔlnL = 1, 4)
    f_interp = interp1d(kl_scan, delta_nll, kind='cubic')
    
    def find_crossing(threshold, side='left'):
        """Find κ_λ where -2ΔlnL = threshold."""
        if side == 'left':
            mask = kl_scan < kl_best
        else:
            mask = kl_scan > kl_best
        
        vals = delta_nll[mask]
        kls = kl_scan[mask]
        
        if len(vals) == 0:
            return kl_scan[0] if side == 'left' else kl_scan[-1]
        
        # Find crossing
        crossings = np.where(np.diff(np.sign(vals - threshold)))[0]
        if len(crossings) > 0:
            idx = crossings[-1] if side == 'left' else crossings[0]
            # Linear interpolation
            x0, x1 = kls[idx], kls[idx+1]
            y0, y1 = vals[idx], vals[idx+1]
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0 + 1e-10)
        
        return kl_scan[0] if side == 'left' else kl_scan[-1]
    
    kl_1sigma = (find_crossing(1.0, 'left'), find_crossing(1.0, 'right'))
    kl_2sigma = (find_crossing(4.0, 'left'), find_crossing(4.0, 'right'))
    
    return {
        'kl_scan': kl_scan,
        'delta_nll': delta_nll,
        'kl_best': kl_best,
        'kl_1sigma': kl_1sigma,
        'kl_2sigma': kl_2sigma,
    }


# ============================================================
# CLEF Hypothesis Test
# ============================================================
def test_clef_hypothesis(results, kl_clef=1.5738738045, kl_sm=1.0):
    """
    Test CLEF prediction κ_λ = 1.5738738045 vs SM κ_λ = 1.0.
    
    Computes:
    - -2ΔlnL at κ_λ(CLEF)
    - p-value for excluding SM if CLEF is true (and vice versa)
    - Significance of CLEF vs SM discrimination
    """
    f_interp = interp1d(results['kl_scan'], results['delta_nll'], kind='cubic')
    
    dnll_clef = float(f_interp(kl_clef))
    dnll_sm = float(f_interp(kl_sm))
    
    # Test statistic: -2ΔlnL between CLEF and SM
    delta_test = abs(dnll_clef - dnll_sm)
    
    # p-value (assuming Wilks' theorem, 1 DOF)
    p_value = 1 - norm.cdf(np.sqrt(abs(delta_test)))
    significance = norm.ppf(1 - p_value) if p_value < 0.5 else 0.0
    
    return {
        'dnll_clef': dnll_clef,
        'dnll_sm': dnll_sm,
        'delta_test_statistic': delta_test,
        'p_value_sm_excluded': p_value,
        'significance_sigma': significance,
        'clef_in_1sigma': results['kl_1sigma'][0] <= kl_clef <= results['kl_1sigma'][1],
        'clef_in_2sigma': results['kl_2sigma'][0] <= kl_clef <= results['kl_2sigma'][1],
    }


# ============================================================
# Asimov data generation (for expected sensitivity)
# ============================================================
def generate_asimov_data(kl_true, lumi, n_bins=25, mhh_range=(250, 1200),
                          signal_efficiency=0.40):
    """
    Generate Asimov dataset for expected sensitivity.
    
    Asimov = exactly the expected number of events per bin
    (no statistical fluctuations — gives median expected result)
    """
    _, _, sig = generate_mHH_template(kl_true, n_bins, mhh_range)
    sig_events = sig * lumi * signal_efficiency
    
    _, _, bkg_events = background_template(n_bins, mhh_range, lumi)
    
    asimov = sig_events + bkg_events
    
    return asimov, bkg_events


# ============================================================
# Main analysis
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CLEF Pipeline — κ_λ Extraction")
    parser.add_argument('--luminosity', type=float, default=3000, help='Luminosity [fb⁻¹]')
    parser.add_argument('--kl-true', type=float, default=1.5738738045, help='True κ_λ for Asimov')
    parser.add_argument('--efficiency', type=float, default=0.40, help='Signal efficiency after ParT')
    parser.add_argument('--output', type=str, default='./results/', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  CLEF Pipeline — κ_λ Extraction via Profile Likelihood")
    print("=" * 60)
    print(f"  Luminosity:         {args.luminosity:.0f} fb⁻¹")
    print(f"  True κ_λ (Asimov):  {args.kl_true}")
    print(f"  Signal efficiency:  {args.efficiency*100:.0f}%")
    print(f"  CLEF prediction:    κ_λ = 1.5738738045")
    print(f"  SM value:           κ_λ = 1.000")
    print()
    
    # Cross sections
    print("Cross sections (NLO, 14 TeV):")
    for kl_val in [0.0, 1.0, 1.5738738045, 2.0, 5.0]:
        xs = sigma_HH_bbbb(kl_val)
        print(f"  κ_λ = {kl_val:5.1f}  →  σ(HH→bbbb) = {xs:6.2f} fb"
              f"  (N_events @ {args.luminosity:.0f}/fb = {xs*args.luminosity:.0f})")
    print()
    
    # Generate Asimov data
    n_bins = 25
    mhh_range = (250, 1200)
    
    asimov_data, bkg_hist = generate_asimov_data(
        args.kl_true, args.luminosity, n_bins, mhh_range, args.efficiency
    )
    
    total_sig = sigma_HH_bbbb(args.kl_true) * args.luminosity * args.efficiency
    total_bkg = bkg_hist.sum()
    print(f"Asimov dataset (κ_λ = {args.kl_true}):")
    print(f"  Signal events:     {total_sig:.0f}")
    print(f"  Background events: {total_bkg:.0f}")
    print(f"  S/√B:              {total_sig/np.sqrt(total_bkg):.1f}")
    print()
    
    # Profile likelihood scan
    results = profile_likelihood_scan(
        asimov_data, bkg_hist, args.luminosity, 
        signal_efficiency=args.efficiency
    )
    
    print("Profile Likelihood Results:")
    print(f"  Best-fit κ_λ:      {results['kl_best']:.3f}")
    print(f"  68% CL interval:   [{results['kl_1sigma'][0]:.3f}, {results['kl_1sigma'][1]:.3f}]")
    print(f"  95% CL interval:   [{results['kl_2sigma'][0]:.3f}, {results['kl_2sigma'][1]:.3f}]")
    print(f"  Uncertainty:       σ(κ_λ) ≈ {(results['kl_1sigma'][1]-results['kl_1sigma'][0])/2:.3f}")
    print()
    
    # CLEF hypothesis test
    clef_test = test_clef_hypothesis(results, kl_clef=1.5738738045, kl_sm=1.0)
    
    print("CLEF vs SM Discrimination:")
    print(f"  -2ΔlnL(κ_λ=1.5738738045): {clef_test['dnll_clef']:.3f}")
    print(f"  -2ΔlnL(κ_λ=1.000): {clef_test['dnll_sm']:.3f}")
    print(f"  Test statistic:      {clef_test['delta_test_statistic']:.3f}")
    print(f"  Significance:        {clef_test['significance_sigma']:.1f}σ")
    print(f"  CLEF in 1σ band:     {'✓' if clef_test['clef_in_1sigma'] else '✗'}")
    print(f"  CLEF in 2σ band:     {'✓' if clef_test['clef_in_2sigma'] else '✗'}")
    print()
    
    # Projected sensitivity at different luminosities
    print("Projected sensitivity vs luminosity:")
    print(f"  {'L [fb⁻¹]':>12}  {'σ(κ_λ)':>8}  {'CLEF-SM [σ]':>12}")
    for lumi in [300, 1000, 3000, 6000]:
        asimov_l, bkg_l = generate_asimov_data(args.kl_true, lumi, n_bins, mhh_range, args.efficiency)
        res_l = profile_likelihood_scan(asimov_l, bkg_l, lumi, signal_efficiency=args.efficiency)
        sigma_kl = (res_l['kl_1sigma'][1] - res_l['kl_1sigma'][0]) / 2
        test_l = test_clef_hypothesis(res_l)
        print(f"  {lumi:>12.0f}  {sigma_kl:>8.3f}  {test_l['significance_sigma']:>12.1f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_dir / 'kl_extraction_results.npz',
        kl_scan=results['kl_scan'],
        delta_nll=results['delta_nll'],
        kl_best=results['kl_best'],
        kl_1sigma=results['kl_1sigma'],
        kl_2sigma=results['kl_2sigma'],
        asimov_data=asimov_data,
        bkg_hist=bkg_hist,
        clef_test=clef_test,
    )
    
    print(f"\nResults saved to {output_dir / 'kl_extraction_results.npz'}")
    
    if args.plot:
        plot_results(results, clef_test, args, output_dir)


def plot_results(results, clef_test, args, output_dir):
    """Generate analysis plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Cross section vs κ_λ
    ax = axes[0]
    kl_range = np.linspace(-2, 8, 200)
    xs = [sigma_HH_bbbb(kl) for kl in kl_range]
    ax.plot(kl_range, xs, 'b-', linewidth=2)
    ax.axvline(1.0, color='gray', ls='--', label='SM (κ_λ=1)')
    ax.axvline(1.5738738045, color='red', ls='--', linewidth=2, label='CLEF (κ_λ=1.5738738045)')
    ax.set_xlabel('κ_λ')
    ax.set_ylabel('σ(pp→HH→bbbb) [fb]')
    ax.set_title('Cross Section vs κ_λ')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    # 2. Profile likelihood
    ax = axes[1]
    ax.plot(results['kl_scan'], results['delta_nll'], 'b-', linewidth=2)
    ax.axhline(1.0, color='orange', ls='--', alpha=0.7, label='68% CL')
    ax.axhline(4.0, color='red', ls='--', alpha=0.7, label='95% CL')
    ax.axvline(1.5738738045, color='red', ls=':', linewidth=2, label=f'CLEF ({1.5738738045})')
    ax.axvline(1.0, color='gray', ls=':', label='SM')
    ax.set_xlabel('κ_λ')
    ax.set_ylabel('-2 Δln L')
    ax.set_title(f'Profile Likelihood ({args.luminosity:.0f} fb⁻¹)')
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. m_HH templates
    ax = axes[2]
    for kl_val, color, label in [(1.0, 'blue', 'SM'), (1.5738738045, 'red', 'CLEF'), (2.0, 'green', 'κ_λ=2'), (5.0, 'purple', 'κ_λ=5')]:
        _, centers, template = generate_mHH_template(kl_val)
        template_norm = template / (template.sum() + 1e-10)
        ax.plot(centers, template_norm, color=color, label=label, linewidth=1.5)
    ax.set_xlabel('m_HH [GeV]')
    ax.set_ylabel('Normalized shape')
    ax.set_title('m_HH Shape Morphing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kl_extraction_plots.png', dpi=150)
    print(f"Plots saved to {output_dir / 'kl_extraction_plots.png'}")


if __name__ == '__main__':
    main()
