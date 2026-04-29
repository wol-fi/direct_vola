# Demo R implementation of the inverse-Gaussian implied-volatility formula.
#
# This is demonstration code for computing total Black-Scholes implied volatility
# from normalized call price and forward log-moneyness. It is not the optimized
# implementation used for the numerical speed results reported in the paper.
#
# Paper:
# An Explicit Solution to Black-Scholes Implied Volatility
# Wolfgang Schadner
# Available at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6649499
#
# Copyright (c) 2026 Wolfgang Schadner.
# All rights reserved.
#
# This code is provided for research and demonstration purposes only, without
# warranty of any kind. It should not be interpreted as production-ready
# numerical software.

# Functions ---------------------------------------------------------------

iv_ig <- function(k, c, n = 30) {
  rt2 <- sqrt(2)
  rt8 <- sqrt(8)
  p0 <- .Machine$double.xmin
  p1 <- 1 - .Machine$double.eps
  twospi <- 2 / sqrt(pi)
  
  one <- function(k, c) {
    if (!is.finite(k) || !is.finite(c)) return(NaN)
    if (c <= 0) return(0)
    if (abs(k) < 1e-12) return(2 * qnorm(min(max(c + 0.5, p0), p1)))
    
    if (k < 0) {
      c <- 1 + exp(-k) * (c - 1)
      k <- -k
    }
    
    c <- min(max(c, p0), p1)
    
    fz <- function(z) {
      pnorm(rt2 * z - k / (2 * rt2 * z)) -
        exp(k) * pnorm(-rt2 * z - k / (2 * rt2 * z)) - c
    }
    
    z <- qnorm(min(max(0.5 * (1 + c), p0), p1)) / rt2
    if (!is.finite(z) || z < 1e-14) z <- 1e-14
    
    f <- fz(z)
    
    if (f < 0) {
      lo <- z
      hi <- 2 * z + 1e-14
      for (i in seq_len(100)) {
        if (fz(hi) >= 0) break
        lo <- hi
        hi <- 2 * hi + 1e-14
      }
    } else {
      hi <- z
      lo <- 0.5 * z
      for (i in seq_len(100)) {
        if (fz(lo) <= 0) break
        hi <- lo
        lo <- 0.5 * lo
      }
    }
    
    z <- 0.5 * (lo + hi)
    
    for (i in seq_len(n)) {
      f <- fz(z)
      if (f < 0) lo <- z else hi <- z
      
      fp <- twospi * exp(0.5 * k - z * z - k * k / (16 * z * z))
      zn <- 0.5 * (lo + hi)
      
      if (fp > 0 && is.finite(fp)) {
        r <- f / fp
        h <- -2 * z + k * k / (8 * z * z * z)
        den <- 2 - r * h
        if (den != 0 && is.finite(den)) {
          cand <- z - 2 * r / den
          if (cand > lo && cand < hi && is.finite(cand)) zn <- cand
        }
      }
      
      z <- zn
    }
    
    rt8 * z
  }
  
  m <- max(length(k), length(c))
  k <- rep(k, length.out = m)
  c <- rep(c, length.out = m)
  vapply(seq_len(m), function(i) one(k[i], c[i]), numeric(1))
}

bs_call <- function(k, v) {
  pnorm(-k / v + 0.5 * v) - exp(k) * pnorm(-k / v - 0.5 * v)
}


# Example -----------------------------------------------------------------

v_true <- c(0.01, seq(0.05, 2, 0.05))
d <- c(0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95)

grid <- expand.grid(v = v_true, d = d)
grid$k <- with(grid, v * (0.5 * v - qnorm(d)))
grid$c <- with(grid, bs_call(k, v))
grid$v_hat <- with(grid, iv_ig(k, c))
grid$abs_err <- abs(grid$v_hat - grid$v)

summary <- data.frame(
  n = nrow(grid),
  mean_abs_err = mean(grid$abs_err),
  max_abs_err = max(grid$abs_err)
)

print(summary)
head(grid)
