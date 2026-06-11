"""Test analysis/stats_utility.py — bootstrap CI, Cohen's d, statistical power, format_pvalue."""

from __future__ import annotations

import numpy as np

from analysis.stats_utility import bootstrap_ci, cohen_d, format_pvalue, statistical_power

# ── format_pvalue ─────────────────────────────────────────────────────────────


def test_format_pvalue_zero_returns_threshold_1000() -> None:
    """p=0.0 con n_perm=1000 → 'p < 0.001'."""
    assert format_pvalue(0.0, 1000) == "p < 0.001"


def test_format_pvalue_zero_returns_threshold_500() -> None:
    """p=0.0 con n_perm=500 → 'p < 0.002'."""
    assert format_pvalue(0.0, 500) == "p < 0.002"


def test_format_pvalue_nonzero_formatted() -> None:
    """p=0.042 → 'p = 0.042'."""
    assert format_pvalue(0.042, 1000) == "p = 0.042"


def test_format_pvalue_nonzero_not_zero() -> None:
    """p=0.001 non deve essere trattato come zero."""
    result = format_pvalue(0.001, 1000)
    assert result == "p = 0.001"


def test_format_pvalue_one() -> None:
    """p=1.0 → 'p = 1.000'."""
    assert format_pvalue(1.0, 1000) == "p = 1.000"


def test_format_pvalue_zero_n_perm_200() -> None:
    """p=0.0 con n_perm=200 → threshold=1/200=0.005."""
    result = format_pvalue(0.0, 200)
    assert result == "p < 0.005"


def test_format_pvalue_returns_string() -> None:
    """format_pvalue restituisce sempre una stringa."""
    assert isinstance(format_pvalue(0.0, 1000), str)
    assert isinstance(format_pvalue(0.5, 1000), str)


# ── bootstrap_ci ──────────────────────────────────────────────────────────────


def test_bootstrap_ci_normal_contains_zero() -> None:
    """CI 95% su 1000 campioni N(0,1) deve contenere la media vera (0)."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(1000)
    lo, hi = bootstrap_ci(values, n_boot=1000, alpha=0.05, random_state=42)
    assert lo < 0.0 < hi, f"CI [{lo:.4f}, {hi:.4f}] non contiene 0"


def test_bootstrap_ci_width_plausible() -> None:
    """CI 95% su N(0,1) n=1000 deve avere ampiezza plausibile (~0.1–0.15)."""
    rng = np.random.default_rng(1)
    values = rng.standard_normal(1000)
    lo, hi = bootstrap_ci(values, n_boot=1000, alpha=0.05, random_state=42)
    width = hi - lo
    assert 0.05 < width < 0.30, f"Ampiezza CI inattesa: {width:.4f}"


def test_bootstrap_ci_tighter_with_more_samples() -> None:
    """CI su campione grande è più stretto di CI su campione piccolo."""
    rng = np.random.default_rng(2)
    small = rng.standard_normal(50)
    large = rng.standard_normal(5000)
    lo_s, hi_s = bootstrap_ci(small, n_boot=500, random_state=42)
    lo_l, hi_l = bootstrap_ci(large, n_boot=500, random_state=42)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_ci_custom_statistic() -> None:
    """bootstrap_ci funziona con statistica personalizzata (mediana)."""
    rng = np.random.default_rng(3)
    values = rng.standard_normal(200)
    lo, hi = bootstrap_ci(values, statistic=np.median, n_boot=500, random_state=42)
    assert lo < hi
    assert -0.5 < lo < 0.5 and -0.5 < hi < 0.5


# ── cohen_d ───────────────────────────────────────────────────────────────────


def test_cohen_d_known_effect() -> None:
    """a=N(0,1), b=N(1,1) → d ≈ 1 (effetto grande)."""
    rng = np.random.default_rng(4)
    a = rng.standard_normal(10000)
    b = rng.standard_normal(10000) + 1.0
    d = cohen_d(a, b)
    assert abs(d - (-1.0)) < 0.05, f"d atteso ≈ -1.0, trovato {d:.4f}"


def test_cohen_d_zero_effect() -> None:
    """Stessa distribuzione → d = 0."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal(1000)
    b = rng.standard_normal(1000)
    # Con n grande e stessa distribuzione, |d| deve essere piccolo
    d = cohen_d(a, b)
    assert abs(d) < 0.15, f"|d| troppo grande per campioni identici: {d:.4f}"


def test_cohen_d_sign() -> None:
    """d è positivo se mean_a > mean_b, negativo se mean_a < mean_b."""
    # Usiamo array con varianza > 0 per evitare s_pooled=0 → inf
    rng = np.random.default_rng(99)
    a_high = rng.standard_normal(100) + 2.0  # media ≈ 2
    b_low = rng.standard_normal(100) + 1.0  # media ≈ 1
    assert cohen_d(a_high, b_low) > 0
    assert cohen_d(b_low, a_high) < 0


def test_cohen_d_identical_groups() -> None:
    """d = 0 se i due campioni sono identici."""
    a = [1.0, 2.0, 3.0]
    assert cohen_d(a, a) == 0.0


def test_cohen_d_medium_effect() -> None:
    """a=N(0,1), b=N(0.5,1) → d ≈ 0.5 (effetto medio)."""
    rng = np.random.default_rng(6)
    a = rng.standard_normal(5000)
    b = rng.standard_normal(5000) + 0.5
    d = cohen_d(a, b)
    assert abs(d - (-0.5)) < 0.05, f"d medio atteso ≈ -0.5, trovato {d:.4f}"


# ── statistical_power ─────────────────────────────────────────────────────────


def test_power_small_effect_low() -> None:
    """Effect size piccolo (d=0.2), n=50 → power < 0.5 (test poco potente)."""
    power = statistical_power(effect_size=0.2, n=50, alpha=0.05)
    assert power < 0.5, f"Power attesa < 0.5 per effetto piccolo, trovata {power:.4f}"


def test_power_large_effect_high() -> None:
    """Effect size grande (d=0.8), n=50 → power > 0.9."""
    power = statistical_power(effect_size=0.8, n=50, alpha=0.05)
    assert power > 0.9, f"Power attesa > 0.9 per effetto grande, trovata {power:.4f}"


def test_power_increases_with_n() -> None:
    """A parità di effect size, power aumenta con n."""
    p_small = statistical_power(0.5, n=20)
    p_large = statistical_power(0.5, n=100)
    assert p_large > p_small


def test_power_in_valid_range() -> None:
    """Power è sempre in [0, 1]."""
    for d in [0.1, 0.5, 1.0, 2.0]:
        for n in [10, 50, 200]:
            p = statistical_power(d, n)
            assert 0.0 <= p <= 1.0, f"Power fuori range: d={d}, n={n}, p={p}"


def test_power_conventional_threshold() -> None:
    """d=0.5, n=67 ≈ minimo per power=0.80 (convezione Cohen 1988)."""
    power = statistical_power(effect_size=0.5, n=67, alpha=0.05)
    assert power >= 0.78, f"Power attesa ≥ 0.78 con n=67, d=0.5: {power:.4f}"


# ── bootstrap_ci_bca ─────────────────────────────────────────────────────────────


def test_bca_matches_known_distribution() -> None:
    """BCa CI su N(0,1) n=1000: deve contenere 0 e width simile a percentile (<1% diff)."""
    from analysis.stats_utility import bootstrap_ci_bca

    rng = np.random.default_rng(99)
    values = rng.standard_normal(1000)

    lo_pct, hi_pct = bootstrap_ci(values, n_boot=2000, alpha=0.05, random_state=42)
    lo_bca, hi_bca = bootstrap_ci_bca(values, n_boot=2000, alpha=0.05, random_state=42)

    # Entrambi devono contenere la media vera (0)
    assert lo_bca < 0.0 < hi_bca, f"BCa CI [{lo_bca:.4f}, {hi_bca:.4f}] non contiene 0"

    # Width BCa non deve discostarsi >1% da percentile per distribuzione gaussiana
    width_pct = hi_pct - lo_pct
    width_bca = hi_bca - lo_bca
    rel_diff = abs(width_bca - width_pct) / max(width_pct, 1e-10)
    assert rel_diff < 0.01, (
        f"BCa width {width_bca:.4f} vs percentile {width_pct:.4f}: "
        f"diff relativa {rel_diff:.4f} > 1%"
    )


def test_bca_lower_bound_lt_upper() -> None:
    """BCa CI: lo < hi sempre."""
    from analysis.stats_utility import bootstrap_ci_bca

    rng = np.random.default_rng(7)
    for _ in range(5):
        values = rng.standard_normal(100)
        lo, hi = bootstrap_ci_bca(values, n_boot=500, random_state=42)
        assert lo < hi, f"BCa lo={lo:.4f} >= hi={hi:.4f}"


# ── hedges_g ─────────────────────────────────────────────────────────────────


def test_hedges_g_correction_factor() -> None:
    """Hedges g < Cohen d (in valore assoluto) per N piccolo."""
    from analysis.stats_utility import hedges_g

    rng = np.random.default_rng(10)
    a_small = rng.standard_normal(15) + 1.0
    b_small = rng.standard_normal(15)

    d = cohen_d(a_small, b_small)
    g = hedges_g(a_small, b_small)

    # g deve essere più piccolo di d in valore assoluto (correzione riduce sovrastima)
    assert abs(g) < abs(d), f"|g|={abs(g):.4f} non < |d|={abs(d):.4f} per N=15"
    # La correzione deve essere nell'ordine atteso (~1-3/(4*30-9)=1-3/111≈0.973)
    expected_correction = 1.0 - 3.0 / (4 * 30 - 9)
    assert abs(g / d - expected_correction) < 0.001, (
        f"Fattore correzione atteso {expected_correction:.4f}, trovato {g / d:.4f}"
    )


def test_hedges_g_converges_to_d_large_n() -> None:
    """Hedges g ≈ Cohen d per N grande (N=5000 per gruppo)."""
    from analysis.stats_utility import hedges_g

    rng = np.random.default_rng(11)
    a_large = rng.standard_normal(5000)
    b_large = rng.standard_normal(5000) + 0.5

    d = cohen_d(a_large, b_large)
    g = hedges_g(a_large, b_large)

    rel_diff = abs(g - d) / max(abs(d), 1e-10)
    assert rel_diff < 0.001, f"g={g:.6f} vs d={d:.6f}: diff relativa {rel_diff:.6f} > 0.1%"


def test_hedges_g_sign_matches_cohen_d() -> None:
    """Segno di g uguale a segno di d."""
    from analysis.stats_utility import hedges_g

    rng = np.random.default_rng(12)
    a = rng.standard_normal(20) + 1.0
    b = rng.standard_normal(20)

    d = cohen_d(a, b)
    g = hedges_g(a, b)
    assert (g > 0) == (d > 0), f"Segno diverso: g={g:.4f}, d={d:.4f}"
