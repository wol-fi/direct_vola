# Explicit Black-Scholes Implied Volatility Demo Code

This repository contains demo Python and R code accompanying the paper:

An Explicit Solution to Black-Scholes Implied Volatility  
Wolfgang Schadner  
[arXiv:2604.24480](https://arxiv.org/abs/2604.24480)

## Scope

This repository contains demonstration implementations of the inverse-Gaussian implied-volatility formula.
It is **not** the fast/native implementation used for the numerical results or speed benchmark reported in the paper.
The higher-level scripts are arranged to make repeated runs and cross-checks easier while keeping the numerical method itself close to the paper.

## Files

- `example_ivol.py` - Python demo script for batched accuracy and timing comparisons against `py_vollib`
- `example_ivol.R` - R demo script for computing implied volatility from normalized inputs and saving a short summary

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

Useful options:

```powershell
python example_ivol.py --runs 5 --reps 500
python example_ivol.py --skip-lbr
python example_ivol.py --output custom_summary.txt
```

R:

```powershell
Rscript example_ivol.R
```

This script writes `iv_r_summary.txt` in the working directory.

You can also choose the R summary output path:

```powershell
Rscript example_ivol.R custom_r_summary.txt
```

## Notes

This code is provided for research and demonstration purposes only. No warranty is provided.
