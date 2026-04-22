# The Russell Index Premium: Replication and Extension

**[→ Full analysis in `project.ipynb`](project.ipynb)**

## Finding

Chang, Hong, and Liskovich ([2015](https://doi.org/10.1093/rfs/hhu041)) identify a ~5% price premium for stocks crossing the Russell 1000/2000 cutoff, using a fuzzy regression discontinuity around rank 1000 by end-of-May market capitalization. Their sample ended in 2012. Since then, passive investing's share of U.S. equity AUM has roughly tripled, from ~15% to ~50%. If index demand is a mechanical driver of prices, the premium should have *grown*.

**It has not.** Over 1996–2024, the reduced-form (ITT) time trend on the cutoff discontinuity is −0.11%/yr for additions (t=−0.56) and −0.06%/yr for deletions (t=−0.64) — directionally consistent with arbitrage capacity scaling alongside passive demand, but statistically indistinguishable from zero. The strong form of the passive-distortion hypothesis is not supported.

![Rolling 3-year ITT estimates of the cutoff discontinuity, 1998–2024](files/figure_itt_rolling.png)

## Selected Results (June returns at the cutoff)

| Effect | Paper LATE (1996–2012) | Replication LATE (1996–2012) | Extension ITT (2015–2024) |
|--------|-----------------------:|-----------------------------:|--------------------------:|
| Addition | +5.0%\*\* (t=2.65) | +2.14% (t=0.44) | +1.24% (t=0.17) |
| Deletion | +5.4%\*\* (t=3.00) | +7.75% (t=1.79) | +0.52% (t=0.19) |

\*\* p<0.05 in the original paper. The replication recovers the correct sign on both effects and a magnitude close to the paper on the deletion side; standard errors are wider than the paper's because the first stage is attenuated (see Limitations). Full Tables 3–6, robustness checks, and period-split ITT estimates are in [`project.ipynb`](project.ipynb).

## Methodology

A fuzzy RD instruments actual Russell 2000 membership (from Bloomberg constituent lists, 1996–2024) with an indicator for predicted end-of-May rank crossing the cutoff, estimated via local linear regression within a 100-rank bandwidth and HC1-robust standard errors. For the post-2007 banded regime, cutoffs are computed from reverse cumulative market cap per the paper's footnote 5. Because the 2015–2024 first stage collapses (F≈0) and inflates fuzzy 2SLS point estimates beyond interpretation, the extension pivots to the reduced-form ITT — a valid causal parameter that does not require a strong first stage (Angrist & Pischke, 2009).

## Limitations

1. **Attenuated first stage.** Russell ranks on proprietary float-adjusted market caps. Bloomberg EQY_FLOAT provides float shares only for the ~3,000 Russell 3000 constituents; eligible non-index stocks near the cutoff fall back to CRSP/Compustat total shares. This mixed-basis ranking injects irreducible noise, yielding α ≈ 0.31 vs. the paper's 0.785. LATE point estimates inherit amplified sampling error, though signs and magnitudes remain broadly consistent.
2. **Post-banding sample collapse.** After 2007, Russell's banding policy cuts the number of switchers near the cutoff to a median of ~13 addition firm-years per reconstitution, driving the post-banding F-statistic to ≈0 and motivating the ITT pivot for the extension.
3. **Short interest coverage.** Compustat short interest begins in 2006; the paper used NYSE/AMEX exchange-level data back to 1993, so the SR outcome is not directly comparable over the pre-2006 window.

## Data

All raw data is proprietary ([WRDS](https://wrds-www.wharton.upenn.edu/) CRSP, Compustat, CCM, Thomson 13F; Bloomberg Terminal for Russell constituent lists and float shares; yfinance for the Russell 2000 index series) and is not included in this repository. The data pipeline — merging, ranking, banding, and outcome construction — is implemented in [`auxiliary/data_processing.py`](auxiliary/data_processing.py); estimation in [`auxiliary/estimation.py`](auxiliary/estimation.py).

## References

- Chang, Y.-C., Hong, H., & Liskovich, I. (2015). Regression Discontinuity and the Price Effects of Stock Market Indexing. *The Review of Financial Studies*, 28(1), 212–246.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Lee, D. S., & Lemieux, T. (2010). Regression Discontinuity Designs in Economics. *Journal of Economic Literature*, 48(2), 281–355.
- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In *Identification and Inference for Econometric Models*. Cambridge University Press.
