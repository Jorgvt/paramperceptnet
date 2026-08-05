# Subjective Database Self-Consistency & Maximum Attainable Correlation

This document explains the concept, formulation, and recalculation of the "maximum attainable correlation" (also referred to as the database self-consistency limit) for the TID2008 and TID2013 datasets.

## Concept: Maximum Attainable Correlation ($\rho_{\text{max}}$)

Subjective image quality assessments are collected from human observers under varying conditions, introducing statistical noise and observer disagreement. Consequently, even a theoretically perfect objective quality metric cannot correlate perfectly ($1.0$) with the subjective Mean Opinion Scores (MOS). 

The maximum attainable correlation ($\rho_{\text{max}}$) defines a performance ceiling—representing the correlation of a single human observer's ratings with the consensus of the group. An objective metric should not exceed this threshold, as doing so would imply it is overfitting to the specific noise patterns of that particular dataset.

## Monte Carlo Simulation Methodology

Since the databases provide the Mean Opinion Score ($MOS_i$) and its corresponding standard error ($STD_i$) for each distorted image $i$:
1. The standard error is related to the individual rating standard deviation ($\sigma_i$) by the number of evaluations per image ($N$):
   $$\sigma_i = \sqrt{N} \cdot STD_i$$
2. A single simulated observer's ratings vector $x$ is drawn from a normal distribution:
   $$x_i \sim \mathcal{N}(MOS_i, (k \cdot STD_i)^2)$$
   where $k = \sqrt{N}$ is the scaling factor representing the translation from standard error to individual variance.
3. The maximum attainable correlation is computed as the average correlation between the simulated observer $x$ and the consensus $MOS$:
   $$\rho_{\text{max}} = \mathbb{E} \left[ \text{corr}(x, MOS) \right]$$

---

## Dataset Parameters & Results

Using the reconstruction script, the following parameters and metrics yield the original values cited in the Parametric PerceptNet paper:

### 1. TID2008
* **Evaluations per Image ($N$)**: $\approx 33$
* **Scaling Factor ($k$)**: $\sqrt{33} \approx 5.7446$
* **Correlation Metric**: Pearson ($PLCC$)
* **Recalculated $\rho_{\text{max}}$**: **$0.8579$** (rounds to **$0.86$**)
  * Observer-sampling Uncertainty (SD): $\pm 0.0055$
  * Monte Carlo Standard Error (SEM, $N=10000$): $\pm 0.000055$

### 2. TID2013
* **Evaluations per Image ($N$)**: $\approx 36$ (Swiss-tournament design)
* **Scaling Factor ($k$)**: $6.0$
* **Correlation Metric**: Spearman ($SROCC$)
* **Recalculated $\rho_{\text{max}}$**: **$0.8289$** (rounds to **$0.83$**)
  * Observer-sampling Uncertainty (SD): $\pm 0.0055$
  * Monte Carlo Standard Error (SEM, $N=10000$): $\pm 0.000055$

### 3. KADID-10k
* **Evaluations per Image ($N$)**: $30$
* **Scaling Factor ($k$)**: $1.0$ (no scaling needed, the database `kadid_dmos.csv` directly provides the rating variance `var` for individual ratings)
* **Correlation Metric**: Pearson ($PLCC$)
* **Recalculated $\rho_{\text{max}}$**: **$0.7800$** (rounds to **$0.78$**)
  * Observer-sampling Uncertainty (SD): $\pm 0.0031$
  * Monte Carlo Standard Error (SEM, $N=10000$): $\pm 0.000031$

---

## How to Recalculate

A python script is provided to run the Monte Carlo simulation and print out the results:

* **Script Location**: [`recalculate_self_consistency.py`](file:///Users/jorgvt/Developer/paramperceptnet/Experiments/SelfConsistency/recalculate_self_consistency.py)
* **Core Function**: [`run_self_consistency`](file:///Users/jorgvt/Developer/paramperceptnet/Experiments/SelfConsistency/recalculate_self_consistency.py#L7-L48)

Run the script using `uv run` in the [`SelfConsistency`](file:///Users/jorgvt/Developer/paramperceptnet/Experiments/SelfConsistency/) directory:

```bash
# For TID2008 (Target: 0.86)
uv run recalculate_self_consistency.py --database tid2008 --method pearson

# For TID2013 (Target: 0.83)
uv run recalculate_self_consistency.py --database tid2013 --method spearman

# For KADID (Target: 0.78)
uv run recalculate_self_consistency.py --database kadid --method pearson
```
