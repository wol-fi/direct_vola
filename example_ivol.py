"""
Demo benchmark for the inverse-Gaussian implied-volatility formula.

This is demo code for a Python-level comparison against py_vollib. It is not
the implementation used for the numerical results or speed benchmark reported
in the paper.

Paper:
An Explicit Solution to Black-Scholes Implied Volatility
Wolfgang Schadner
Available at: https://arxiv.org/abs/2604.24480

Copyright (c) 2026 Wolfgang Schadner.
All rights reserved.

No warranty is provided. This code is for research and demonstration purposes
only and should not be interpreted as production-ready numerical software.
"""

from __future__ import annotations

import argparse
import gc
from math import erfc
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
from py_vollib.black.implied_volatility import implied_volatility as lbr

DEFAULT_RUNS = 3
DEFAULT_REPS = 200
DEFAULT_OUTPUT = "iv_accuracy_speed_summary.txt"
P0 = np.nextafter(0.0, 1.0)
P1 = np.nextafter(1.0, 0.0)


@njit
def ncdf(x):
    return 0.5 * erfc(-x / 1.4142135623730951)


@njit
def ndtri(p):
    if p <= P0:
        p = P0
    if p >= P1:
        p = P1

    a0 = -3.969683028665376e01
    a1 = 2.209460984245205e02
    a2 = -2.759285104469687e02
    a3 = 1.383577518672690e02
    a4 = -3.066479806614716e01
    a5 = 2.506628277459239e00
    b0 = -5.447609879822406e01
    b1 = 1.615858368580409e02
    b2 = -1.556989798598866e02
    b3 = 6.680131188771972e01
    b4 = -1.328068155288572e01
    c0 = -7.784894002430293e-03
    c1 = -3.223964580411365e-01
    c2 = -2.400758277161838e00
    c3 = -2.549732539343734e00
    c4 = 4.374664141464968e00
    c5 = 2.938163982698783e00
    d0 = 7.784695709041462e-03
    d1 = 3.224671290700398e-01
    d2 = 2.445134137142996e00
    d3 = 3.754408661907416e00

    if p < 0.02425:
        q = np.sqrt(-2.0 * np.log(p))
        x = (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
            ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
        )
    elif p <= 0.97575:
        q = p - 0.5
        r = q * q
        x = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q / (
            (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)
        )
    else:
        q = np.sqrt(-2.0 * np.log1p(-p))
        x = -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
            ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
        )

    for _ in range(2):
        e = ncdf(x) - p
        u = e * 2.5066282746310002 * np.exp(0.5 * x * x)
        x -= u / (1.0 + 0.5 * x * u)

    return x


@njit
def bs_c(k, v):
    d1 = -k / v + 0.5 * v
    d2 = d1 - v
    return ncdf(d1) - np.exp(k) * ncdf(d2)


@njit
def fz(k, z, c):
    if z <= 0.0:
        return -c
    t = k / (2.8284271247461903 * z)
    return (
        ncdf(1.4142135623730951 * z - t)
        - np.exp(k) * ncdf(-1.4142135623730951 * z - t)
        - c
    )


@njit
def fpz(k, z):
    return 1.1283791670955126 * np.exp(0.5 * k - z * z - k * k / (16.0 * z * z))


@njit
def z_start(c):
    z = ndtri(0.5 * (1.0 + c)) / 1.4142135623730951
    if z < 1e-12:
        z = 1e-12
    return z


@njit
def z_root(k, c):
    if c <= P0:
        c = P0
    if c >= P1:
        c = P1

    z = z_start(c)
    f = fz(k, z, c)

    if f < 0.0:
        lo = z
        hi = 2.0 * z + 1e-12
        for _ in range(100):
            if fz(k, hi, c) >= 0.0:
                break
            lo = hi
            hi = 2.0 * hi + 1e-12
    else:
        hi = z
        lo = 0.5 * z
        for _ in range(100):
            if fz(k, lo, c) <= 0.0:
                break
            hi = lo
            lo = 0.5 * lo

    z = 0.5 * (lo + hi)

    for _ in range(30):
        f = fz(k, z, c)

        if f < 0.0:
            lo = z
        else:
            hi = z

        fp = fpz(k, z)
        zn = 0.5 * (lo + hi)

        if fp > 0.0 and np.isfinite(fp):
            r = f / fp
            h = -2.0 * z + k * k / (8.0 * z * z * z)
            den = 2.0 - r * h

            if den != 0.0 and np.isfinite(den):
                cand = z - 2.0 * r / den
                if cand > lo and cand < hi and np.isfinite(cand):
                    zn = cand

        z = zn

    return z


@njit
def iv_fig_numba(k, c):
    if abs(k) < 1e-12:
        return 2.0 * ndtri(0.5 * (1.0 + c))
    if k < 0.0:
        c = 1.0 + np.exp(-k) * (c - 1.0)
        k = -k
    return 2.8284271247461903 * z_root(k, c)


@njit
def iv_fig_batch(ks, cs):
    out = np.empty_like(ks)
    for i in range(ks.size):
        out[i] = iv_fig_numba(ks[i], cs[i])
    return out


def make_cases():
    vs = np.r_[0.01, 0.05 * np.arange(1, 41)]
    ds = np.array([0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95], dtype=np.float64)
    grid_v, grid_d = np.meshgrid(vs, ds, indexing="ij")

    v = grid_v.ravel().astype(np.float64)
    d = grid_d.ravel().astype(np.float64)
    k = np.array([float(vv * (0.5 * vv - ndtri(float(dd)))) for vv, dd in zip(v, d)], dtype=np.float64)
    K = np.exp(k)
    c = np.array([float(bs_c(float(kk), float(vv))) for kk, vv in zip(k, v)], dtype=np.float64)

    return {
        "k": k,
        "K": K,
        "c": c,
        "v": v,
        "d": d,
    }


def iv_lbr_batch(cases):
    c = cases["c"]
    K = cases["K"]
    out = np.empty_like(c)
    for i in range(c.size):
        out[i] = float(lbr(float(c[i]), 1.0, float(K[i]), 0.0, 1.0, "c"))
    return out


def iv_fig_batch_py(cases):
    return iv_fig_batch(cases["k"], cases["c"])


def acc(name, vals, true):
    e = np.abs(vals - true)
    return {
        "method": name,
        "mean_abs_err": float(e.mean()),
        "max_abs_err": float(e.max()),
        "nan": int(np.isnan(vals).sum()),
    }


def speed(name, func, cases, runs, reps):
    func(cases)

    gc.disable()
    ts = []
    n = reps * cases["k"].size

    try:
        for _ in range(runs):
            t0 = perf_counter()
            checksum = 0.0
            for _ in range(reps):
                checksum += float(func(cases).sum())
            ts.append(perf_counter() - t0)
            if not np.isfinite(checksum):
                raise RuntimeError(f"{name} produced a non-finite checksum.")
    finally:
        gc.enable()

    ts = np.array(ts, dtype=np.float64)
    return {
        "method": name,
        "seconds_median": float(np.median(ts)),
        "seconds_mean": float(ts.mean()),
        "us_per_eval_median": float(1e6 * np.median(ts) / n),
        "us_per_eval_mean": float(1e6 * ts.mean() / n),
    }


def format_table(rows, columns):
    widths = {}
    for col in columns:
        values = [col] + [str(row[col]) for row in rows]
        widths[col] = max(len(v) for v in values)

    header = " ".join(col.ljust(widths[col]) for col in columns)
    body = [
        " ".join(str(row[col]).ljust(widths[col]) for col in columns)
        for row in rows
    ]
    return "\n".join([header] + body)


def summarize(cases, runs, reps, with_lbr):
    true = cases["v"]

    # Warm up the jitted path outside the timing loop.
    iv_fig_batch_py({key: value[:1] for key, value in cases.items()})

    accuracy_rows = []
    speed_rows = []

    fig_vals = iv_fig_batch_py(cases)
    accuracy_rows.append(acc("F_IG_numba_batch", fig_vals, true))
    speed_rows.append(speed("F_IG_numba_batch", iv_fig_batch_py, cases, runs, reps))

    if with_lbr:
        lbr_vals = iv_lbr_batch(cases)
        accuracy_rows.insert(0, acc("LBR_vollib", lbr_vals, true))
        speed_rows.insert(0, speed("LBR_vollib", iv_lbr_batch, cases, runs, reps))

    return accuracy_rows, speed_rows


def build_report(accuracy_rows, speed_rows):
    accuracy_columns = ["method", "mean_abs_err", "max_abs_err", "nan"]
    speed_columns = [
        "method",
        "seconds_median",
        "seconds_mean",
        "us_per_eval_median",
        "us_per_eval_mean",
    ]

    parts = [
        "Accuracy summary",
        format_table(accuracy_rows, accuracy_columns),
        "",
        "Speed summary",
        format_table(speed_rows, speed_columns),
    ]
    return "\n".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo benchmark for the inverse-Gaussian implied-volatility formula."
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of timing runs.")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Repetitions per timing run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output path for the text summary.",
    )
    parser.add_argument(
        "--skip-lbr",
        action="store_true",
        help="Skip the slower py_vollib comparison and benchmark only the demo formula.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = make_cases()
    accuracy_rows, speed_rows = summarize(
        cases=cases,
        runs=args.runs,
        reps=args.reps,
        with_lbr=not args.skip_lbr,
    )

    report = build_report(accuracy_rows, speed_rows)
    print(report)

    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
