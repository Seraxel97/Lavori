"""Analisi statistica calibration: bootstrap CI, sign test, accuracy rate."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .feedback_simulator import FeedbackResult


@dataclass
class CalibrationReport:
    n_trades: int
    prediction_accuracy_rate: float   # % trade con sign(predicted) == sign(actual)
    mean_abs_error: float
    mean_error: float                  # bias (positivo = sovrastima)
    bootstrap_CI_95: tuple[float, float]
    sign_test_p: float                 # p-value sign test vs H0: acc = 0.5
    calibration_pass: bool             # sign_test_p < 0.05 AND accuracy > 0.6


def _bootstrap_mean(errors: list[float], n_iter: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap CI 95% sulla media degli errori assoluti."""
    rng = random.Random(seed)
    n = len(errors)
    if n == 0:
        return (0.0, 0.0)
    means: list[float] = []
    for _ in range(n_iter):
        sample = [rng.choice(errors) for _ in range(n)]
        means.append(sum(abs(x) for x in sample) / n)
    means.sort()
    lo = means[int(0.025 * n_iter)]
    hi = means[int(0.975 * n_iter)]
    return (round(lo, 4), round(hi, 4))


def _sign_test_p(n_correct: int, n_total: int) -> float:
    """
    Binomial sign test: H0 = accuracy = 0.5 (random baseline).
    Usa approssimazione normale (z-test) per n ≥ 10.
    """
    if n_total == 0:
        return 1.0
    if n_total < 10:
        # exact binomial CDF (piccolo campione)
        from math import comb
        p_val = sum(comb(n_total, k) * (0.5 ** n_total) for k in range(n_correct, n_total + 1))
        return round(min(p_val * 2, 1.0), 4)  # two-sided
    z = (n_correct - 0.5 * n_total) / math.sqrt(0.25 * n_total)
    # Approssimazione CDF normale standard (formula di Zelen & Severo 1964)
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p_one_tail = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z ** 2) * poly * t
    p_two = min(2 * p_one_tail, 1.0)
    return round(p_two, 4)


def analyze(results: list[FeedbackResult]) -> CalibrationReport:
    """Calcola metriche di calibration da lista di FeedbackResult."""
    n = len(results)
    if n == 0:
        return CalibrationReport(
            n_trades=0,
            prediction_accuracy_rate=0.0,
            mean_abs_error=0.0,
            mean_error=0.0,
            bootstrap_CI_95=(0.0, 0.0),
            sign_test_p=1.0,
            calibration_pass=False,
        )

    errors = [r.prediction_error for r in results]
    abs_errors = [abs(e) for e in errors]

    # sign accuracy: predicted > 0 AND actual > 0, oppure entrambi ≤ 0
    n_correct = sum(
        1 for r in results
        if (r.predicted_roi > 0) == (r.actual_roi > 0)
    )
    accuracy = n_correct / n
    mae = sum(abs_errors) / n
    mean_err = sum(errors) / n
    ci = _bootstrap_mean(errors, n_iter=1000)
    p_val = _sign_test_p(n_correct, n)
    passes = p_val < 0.05 and accuracy > 0.6

    return CalibrationReport(
        n_trades=n,
        prediction_accuracy_rate=round(accuracy, 4),
        mean_abs_error=round(mae, 4),
        mean_error=round(mean_err, 4),
        bootstrap_CI_95=ci,
        sign_test_p=p_val,
        calibration_pass=passes,
    )
