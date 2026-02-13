# Regression Discontinuity and the Price Effects of Stock Market Indexing

[![Run Notebook](https://github.com/<username>/russell-rd-replication/actions/workflows/run-notebook.yml/badge.svg)](https://github.com/<username>/russell-rd-replication/actions/workflows/run-notebook.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Replication Project

The notebook `project.ipynb` contains a replication project for the [ECON 481: Economics Data Science](https://catalog.registrar.washington.edu/course/ECON/481) class at the University of Washington. It replicates the main results from the following paper:

> Chang, Y.-C., Hong, H., & Liskovich, I. (2015). [Regression Discontinuity and the Price Effects of Stock Market Indexing](https://doi.org/10.1093/rfs/hhu041). *The Review of Financial Studies*, 28(1), 212–246.

Chang et al. (2015) use a fuzzy regression discontinuity design to estimate the causal price effects of stock market indexing. The Russell 1000 and Russell 2000 indexes comprise the first 1,000 and next 2,000 largest U.S. firms ranked by market capitalization. Because the indexes are value-weighted, stocks just below the 1,000 cutoff receive significantly higher index weight — and thus more passive buying pressure — than stocks just above. Exploiting this discontinuity over the period 1996–2012, the authors find symmetric addition and deletion effects of approximately 5%, estimate a price elasticity of demand around −1.5, and document that demand curves have become more elastic over time as arbitrage capacity has grown.

## Project Structure

```
├── auxiliary/          # Helper functions for data processing and estimation
├── data/               # Raw and processed datasets
├── files/              # Output figures and tables
├── tests/              # Unit tests for auxiliary functions
├── project.ipynb       # Main project notebook
├── environment.yml     # Conda environment specification
└── pyproject.toml      # Project configuration and linting settings
```

## Reproducibility

To reproduce the results, clone this repository and create the conda environment:

```bash
$ conda env create -f environment.yml
$ conda activate russell-rd
$ jupyter lab
```

Then open and run `project.ipynb`.

## Data Sources

- **CRSP US Stock Database** — Stock prices, returns, shares outstanding, and trading volume
- **Compustat** — Quarterly shares outstanding (CSHOQ), earnings report dates (RDQ), and firm fundamentals
- **Russell/FTSE Russell** — Annual constituent lists for the Russell 1000 and Russell 2000 (1996–2012)

Data is accessed through [WRDS (Wharton Research Data Services)](https://wrds-www.wharton.upenn.edu/).
