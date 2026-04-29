# Explicit Black-Scholes Implied Volatility Demo Code

This repository contains the demo Python code accompanying the paper:

An Explicit Solution to Black-Scholes Implied Volatility  
Wolfgang Schadner  
[arXiv:2604.24480](https://arxiv.org/abs/2604.24480)

## Scope

This is demonstration code for a Python-level comparison against `py_vollib`.
It is **not** the fast/native implementation used for the numerical results or speed benchmark reported in the paper.

## Files

- `example_ivol.py` - demo script for accuracy and timing comparisons
- `iv_accuracy_speed_summary.txt` - example output from one benchmark run

## Requirements

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python example_ivol.py
```

The script writes `iv_accuracy_speed_summary.txt` in the working directory.

## Notes

This code is provided for research and demonstration purposes only. No warranty is provided.
