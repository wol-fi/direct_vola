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

import gc
from time import perf_counter
from math import erfc
import numpy as np
import pandas as pd
from numba import njit
from py_vollib.black.implied_volatility import implied_volatility as lbr

RUNS = 1
REPS = 10
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

    a0 = -3.969683028665376e+01
    a1 = 2.209460984245205e+02
    a2 = -2.759285104469687e+02
    a3 = 1.383577518672690e+02
    a4 = -3.066479806614716e+01
    a5 = 2.506628277459239e+00
    b0 = -5.447609879822406e+01
    b1 = 1.615858368580409e+02
    b2 = -1.556989798598866e+02
    b3 = 6.680131188771972e+01
    b4 = -1.328068155288572e+01
    c0 = -7.784894002430293e-03
    c1 = -3.223964580411365e-01
    c2 = -2.400758277161838e+00
    c3 = -2.549732539343734e+00
    c4 = 4.374664141464968e+00
    c5 = 2.938163982698783e+00
    d0 = 7.784695709041462e-03
    d1 = 3.224671290700398e-01
    d2 = 2.445134137142996e+00
    d3 = 3.754408661907416e+00

    if p < 0.02425:
        q = np.sqrt(-2.0 * np.log(p))
        x = (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)
    elif p <= 0.97575:
        q = p - 0.5
        r = q * q
        x = (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5) * q / (((((b0*r+b1)*r+b2)*r+b3)*r+b4)*r+1.0)
    else:
        q = np.sqrt(-2.0 * np.log1p(-p))
        x = -(((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)

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
    return ncdf(1.4142135623730951 * z - t) - np.exp(k) * ncdf(-1.4142135623730951 * z - t) - c

@njit
def fpz(k, z):
    return 1.1283791670955126 * np.exp(0.5 * k - z*z - k*k / (16.0*z*z))

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
            h = -2.0 * z + k*k / (8.0*z*z*z)
            den = 2.0 - r*h

            if den != 0.0 and np.isfinite(den):
                cand = z - 2.0*r/den
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

def make_cases():
    vs = np.r_[0.01, 0.05 * np.arange(1, 41)]
    ds = np.array([0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95])
    xs = []

    for v in vs:
        for d in ds:
            k = float(v * (0.5 * v - ndtri(float(d))))
            K = float(np.exp(k))
            c = float(bs_c(k, float(v)))
            xs.append((k, K, c, float(v), float(d)))

    return xs

def iv_lbr(x):
    k, K, c, v, d = x
    return float(lbr(c, 1.0, K, 0.0, 1.0, "c"))

def iv_fig(x):
    k, K, c, v, d = x
    return float(iv_fig_numba(k, c))

def acc(name, vals, true):
    e = np.abs(vals - true)
    return {
        "method": name,
        "mean_abs_err": float(e.mean()),
        "max_abs_err": float(e.max()),
        "nan": int(np.isnan(vals).sum())
    }

def speed(name, f, xs):
    for x in xs:
        f(x)

    gc.disable()
    ts = []
    n = REPS * len(xs)

    try:
        for _ in range(RUNS):
            t0 = perf_counter()
            s = 0.0

            for _ in range(REPS):
                for x in xs:
                    s += f(x)

            ts.append(perf_counter() - t0)
    finally:
        gc.enable()

    ts = np.array(ts)

    return {
        "method": name,
        "seconds_median": float(np.median(ts)),
        "seconds_mean": float(ts.mean()),
        "us_per_eval_median": float(1e6 * np.median(ts) / n),
        "us_per_eval_mean": float(1e6 * ts.mean() / n)
    }

xs = make_cases()
true = np.array([x[3] for x in xs], dtype=np.float64)

iv_fig(xs[0])
iv_lbr(xs[0])

y_lbr = np.array([iv_lbr(x) for x in xs])
y_fig = np.array([iv_fig(x) for x in xs])

acc_df = pd.DataFrame([
    acc("LBR_vollib", y_lbr, true),
    acc("F_IG_numba", y_fig, true)
])

spd_df = pd.DataFrame([
    speed("LBR_vollib", iv_lbr, xs),
    speed("F_IG_numba", iv_fig, xs)
])

pd.set_option("display.precision", 16)

print("Accuracy summary")
print(acc_df.to_string(index=False))
print()
print("Speed summary")
print(spd_df.to_string(index=False))

with open("iv_accuracy_speed_summary.txt", "w", encoding="utf-8") as f:
    f.write("Accuracy summary\n")
    f.write(acc_df.to_string(index=False))
    f.write("\n\nSpeed summary\n")
    f.write(spd_df.to_string(index=False))

print("Saved: iv_accuracy_speed_summary.txt")