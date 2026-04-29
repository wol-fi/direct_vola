# Explicit Black-Scholes Implied Volatility Demo Code

This repository contains demo Python and R code accompanying the paper:

An Explicit Solution to Black-Scholes Implied Volatility  
Wolfgang Schadner  
[arXiv:2604.24480](https://arxiv.org/abs/2604.24480)

## Scope

This repository contains demonstration implementations of the inverse-Gaussian implied-volatility formula.
It is **not** the fast/native implementation used for the numerical results or speed benchmark reported in the paper.

## Files

- `example_ivol.py` - Python demo script for accuracy and timing comparisons against `py_vollib`
- `example_ivol.R` - R demo script for computing implied volatility from normalized inputs

## Requirements

For Python:

```powershell
python -m pip install -r requirements.txt
```

For R:

The R script uses base R functions only and does not require additional packages.

## Run

Python:

```powershell
python example_ivol.py
```

This script writes `iv_accuracy_speed_summary.txt` in the working directory.

R:

```powershell
Rscript example_ivol.R
```

## Notes

This code is provided for research and demonstration purposes only. No warranty is provided.
