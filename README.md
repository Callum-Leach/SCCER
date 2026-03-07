# Scenario-coupled climate extreme-value regression (SCCER)

This repository contains the code, data, and figures supporting the paper:

`Changes in Extreme Temperatures of the Earth's Desert Regions over the Next 100 Years.`

The project investigates how extreme near-surface air temperatures (tas) may change in major desert regions over the next century using output from CMIP6 global climate models (GCMs). In particular, the study estimates the change in the 100-year return value of annual temperature extremes between 2025 and 2125.

To characterise changes in extremes, the repository implements scenario-coupled non-stationary Generalised Extreme Value (GEV) regression models considering a range of different parametric forms. These models allow GEV parameters to vary over time while ensuring that different emissions scenarios share a common distribution of extremes at the start of the simulation period. Model parameters are estimated using Bayesian inference with MCMC, enabling full uncertainty quantification.

**Repo overview**
The repository is structured as follows:
**Data:** Main datasets used in the analysis (CMIP6-derived regional extremes).
**src/nonstationary_extremes:** Core software for fitting non-stationary GEV models using Bayesian inference and MCMC.
**Plots:** Generated figures for the paper and supplementary material.

## Authors

| name | Email |
| --- | --- |
|Callum Leach  | callumleach31@gmail.com |
|Philip Jonathan  | ygraigarw@gmail.com |