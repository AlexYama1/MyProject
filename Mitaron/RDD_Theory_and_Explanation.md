# Regression Discontinuity Design: Theory and Implementation Guide

## Table of Contents
1. [What is Regression Discontinuity Design?](#what-is-regression-discontinuity-design)
2. [Fundamental Theory](#fundamental-theory)
3. [Types of RDD](#types-of-rdd)
4. [Mathematical Framework](#mathematical-framework)
5. [Estimation Methods](#estimation-methods)
6. [Validity Checks](#validity-checks)
7. [Your Implementation Explained](#your-implementation-explained)
8. [Code Walkthrough](#code-walkthrough)
9. [Interpreting Results](#interpreting-results)
10. [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)

---

## What is Regression Discontinuity Design?

### The Core Idea

Regression Discontinuity Design (RDD) is a **quasi-experimental research design** used to estimate causal effects when treatment assignment is determined by whether an observable variable (the "running variable") crosses a threshold (the "cutoff").

**Real-world analogy**: Imagine a scholarship program that gives financial aid to all students who score 70 or above on an entrance exam. Students scoring 69 don't get the scholarship, but students scoring 70 do. The key insight: students scoring 69 vs. 70 are probably very similar in ability, so comparing their outcomes lets us estimate the causal effect of the scholarship.

### Why RDD is Powerful

1. **Quasi-random assignment**: Near the cutoff, treatment assignment is "as-if random"
2. **No confounding**: Predetermined characteristics should be continuous at the cutoff
3. **Transparent assumptions**: The continuity assumption is testable
4. **Credible causal inference**: Often considered the "gold standard" among quasi-experimental methods

### Key Components

Every RDD has three essential elements:

1. **Running Variable (X)**: The variable that determines treatment assignment
   - *Your case*: Time (month when treatment was received)

2. **Cutoff (c)**: The threshold that determines treatment
   - *Your case*: January 2022 (when the policy was implemented)

3. **Treatment (D)**: The intervention being studied
   - *Your case*: Post-policy period (eligibility for free healthcare)

---

## Fundamental Theory

### The Identification Strategy

RDD relies on a **local randomization** assumption:

> **Key Assumption**: Units just below and just above the cutoff are comparable in all respects except treatment assignment.

This means:
- A student scoring 69.9 is essentially identical to one scoring 70.1
- The only difference is that one received treatment (crossed the threshold) and the other didn't
- Therefore, differences in outcomes can be attributed to the treatment

### The Continuity Assumption

More formally, RDD requires that **potential outcomes are continuous functions of the running variable** at the cutoff.

**Potential outcomes framework**:
- Y₁(x) = outcome if treated, given running variable = x
- Y₀(x) = outcome if not treated, given running variable = x

**Continuity assumption**:
```
lim[x→c⁻] E[Y₀(x)] = lim[x→c⁺] E[Y₀(x)]
lim[x→c⁻] E[Y₁(x)] = lim[x→c⁺] E[Y₁(x)]
```

In plain English: If there were no treatment, the relationship between the outcome and running variable would be smooth (no jump) at the cutoff.

### What This Means Visually

Imagine plotting outcome Y against running variable X:
- If there were no treatment, we'd see a smooth line through the cutoff
- The treatment creates a **discontinuity** (a jump) at the cutoff
- The size of this jump = the treatment effect

```
Outcome (Y)
    |
    |         ●●●●● ← Treated (X ≥ c)
    |       ●●
    |     ●●  ↑ Treatment Effect (τ)
    |   ●●    ↓
    | ●●●●●● ← Control (X < c)
    |__________|_________________
              c (cutoff)     Running Variable (X)
```

---

## Types of RDD

### 1. Sharp RDD

**Definition**: Treatment is a deterministic function of the running variable.

**Rule**: D = 1 if X ≥ c, else D = 0

**Example**: Scholarship given to ALL students scoring ≥70 (no exceptions)

**Your case**: This is a **Sharp RDD**
- If time ≥ January 2022 → post-policy period (treatment)
- If time < January 2022 → pre-policy period (control)

### 2. Fuzzy RDD

**Definition**: The probability of treatment changes discontinuously at the cutoff, but treatment is not deterministic.

**Rule**: P(D=1|X=c⁺) > P(D=1|X=c⁻), but not everyone above c is treated

**Example**: Scholarship offered to students scoring ≥70, but some decline it

**Estimation**: Uses instrumental variables (IV) approach
- The cutoff becomes an instrument for actual treatment receipt

### 3. Time-Based RDD (Your Case)

**Definition**: A variant where the running variable is time rather than a cross-sectional variable.

**Unique features**:
- Treatment assignment changes at a specific date
- Often combined with age restrictions to create age-homogeneous cohorts
- Blurs the line between RDD and Interrupted Time Series (ITS)

**Your innovation**: Using an age-homogeneous cohort (±1 month around age 17) to implement a time-based cutoff
- Holds age constant across the sample
- Running variable becomes time (when treatment was received)
- Exploits policy implementation date as the cutoff

---

## Mathematical Framework

### The Basic RDD Model

The simplest RDD specification is:

```
Y = β₀ + τ·D + β₁·(X - c) + ε
```

Where:
- Y = outcome variable (e.g., medical expenses)
- D = treatment indicator (1 if X ≥ c, else 0)
- X = running variable
- c = cutoff value
- τ = **treatment effect** (the parameter we care about!)
- (X - c) = centered running variable
- ε = error term

**Interpretation of τ**: The average treatment effect at the cutoff

### Allowing Different Slopes

A more flexible model allows the relationship between Y and X to differ on either side of the cutoff:

```
Y = β₀ + τ·D + β₁·(X - c) + β₂·[D × (X - c)] + ε
```

The new term **D × (X - c)** allows:
- Different slope before the cutoff: β₁
- Different slope after the cutoff: β₁ + β₂

**Why this matters**: The relationship between outcome and running variable may change after treatment.

**Visual representation**:
```
Outcome
    |       ● ● ● ●     ← Slope = β₁ + β₂
    |     ●●    ↑
    |   ●      τ (jump at cutoff)
    | ● ● ●    ↓
    |● ●       ← Slope = β₁
    |___|_________________
        c   Running Variable
```

### Higher-Order Polynomials

Some researchers use quadratic or cubic terms:

**Quadratic**:
```
Y = β₀ + τ·D + β₁·X + β₂·X² + β₃·(D×X) + β₄·(D×X²) + ε
```

**Cubic**:
```
Y = β₀ + τ·D + β₁·X + β₂·X² + β₃·X³ + β₄·(D×X) + β₅·(D×X²) + β₆·(D×X³) + ε
```

**Important warning**: Gelman & Imbens (2019) show that high-order polynomials can:
- Create spurious results
- Be sensitive to outliers
- Produce misleading confidence intervals

**Recommendation**: Stick with linear specifications when possible.

---

## Estimation Methods

### 1. Local Linear Regression (Your Method)

**Core idea**: Only use observations close to the cutoff (within a bandwidth h).

**Why local?**
- Most credible comparisons happen near the cutoff
- Reduces bias from functional form misspecification
- Implements the "local randomization" assumption literally

**Bandwidth (h)**: The window around the cutoff
- Example: h = 1 year means using observations from [c - 1, c + 1]
- Larger h → more data, less variance, but more bias
- Smaller h → less bias, but more variance, less data

### 2. Kernel Weighting

Instead of hard cutoffs (use if |X - c| < h, else exclude), kernel weighting gives **more weight to observations closer to the cutoff**.

**Triangular kernel** (your choice):
```
w(x) = (1 - |x - c| / h)  if |x - c| < h
       0                   otherwise
```

**Properties**:
- Maximum weight (w = 1) at the cutoff
- Linearly decreases as you move away
- Zero weight outside bandwidth

**Visual**:
```
Weight
  1.0 |    /\
      |   /  \
  0.5 |  /    \
      | /      \
  0.0 |/_______\________
      c-h  c  c+h    X
```

**Why triangular?**
- Optimal boundary bias properties
- Simple to implement
- Recommended in the literature (Fan & Gijbels 1996)

### 3. Weighted Least Squares (WLS)

Your code uses **WLS** regression with triangular kernel weights:

```python
model = smf.wls(
    formula='outcome ~ post + time_centered + post:time_centered',
    data=df,
    weights=df['triangular_weight']
)
```

**What this does**:
1. Observations closer to cutoff get more influence on the estimates
2. Automatically implements local linear regression
3. Standard errors account for weighting

### 4. Clustered Standard Errors

**The problem**: Your data has multiple observations per patient (panel structure)
- Same patient appears in multiple months
- Errors are correlated within patient
- Standard errors are too small without clustering

**The solution**: Cluster standard errors by patient ID

```python
model.fit(cov_type='cluster', cov_kwds={'groups': df['patient_id']})
```

**Effect**:
- Accounts for within-patient correlation
- Standard errors increase (become more conservative)
- Confidence intervals widen
- More valid inference

---

## Validity Checks

### Why Validity Checks Matter

The RDD design is only valid if our assumptions hold. We can't directly test the continuity assumption (we can't observe the counterfactual), but we can perform **indirect tests** that would fail if our assumptions were violated.

### 1. Covariate Balance Test

**Logic**: Predetermined characteristics (things that couldn't have been affected by treatment) should be continuous at the cutoff.

**Why**: If there's a jump in sex, income, or family type at the cutoff, it suggests:
- Selection bias
- Manipulation of the running variable
- Violation of the "as-if random" assumption

**How to test**:
- Run the same RDD regression on each covariate
- Treatment effect should be zero (or close to zero)
- p-values should be > 0.05

**Your results**:
```
Covariate        | Coefficient | p-value | Verdict
-----------------|-------------|---------|--------
Sex              | -0.093      | 0.538   | ✓ PASS
Family Type      | -0.305      | 0.061   | ✓ PASS
Occupation       | 0.085       | 0.470   | ✓ PASS
Income Category  | 0.008       | 0.926   | ✓ PASS
```

**Interpretation**: No predetermined characteristics jump at the cutoff → Good!

### 2. Placebo Cutoff Tests

**Logic**: If we see a treatment effect at fake cutoffs where there was no actual policy change, our results might be spurious.

**How to test**:
- Run RDD at fake cutoffs (e.g., ±3 months, ±6 months from true cutoff)
- Should find no treatment effect (τ ≈ 0, p > 0.05)

**Your results**:
```
Placebo Cutoff   | Coefficient | p-value | Verdict
-----------------|-------------|---------|--------
-6 months        | -142        | 0.955   | ✓ PASS
-3 months        | -1,547      | 0.560   | ✓ PASS
+3 months        | 3,393       | 0.193   | ✓ PASS
+6 months        | 88          | 0.968   | ✓ PASS
```

**Interpretation**: No effects at fake cutoffs → Good!

### 3. Density Test (McCrary Test)

**Logic**: If individuals can manipulate the running variable to get treatment, we'll see:
- "Bunching" just above the cutoff
- A discontinuous jump in the density of the running variable

**Example of manipulation**:
- If students know the scholarship cutoff is 70
- They might retake exams until they score ≥70
- We'd see too many scores of exactly 70, and too few scores of 69

**How to test**:
1. Create a histogram of the running variable
2. Visual inspection: Is there a sharp jump at the cutoff?
3. Formal test: McCrary (2008) test for discontinuity in density

**Your case**: Time cannot be manipulated
- Patients can't change when they receive treatment (time flows naturally)
- This test is more of a data quality check
- Looking for smooth distribution of observations over time

**What to look for**:
```
# Observations
    |
 20 |  ▄▄       ▄▄▄     ← Should be relatively smooth
 15 |▄▄▄▄▄   ▄▄▄▄▄▄
 10 |▄▄▄▄▄▄ ▄▄▄▄▄▄▄
  5 |▄▄▄▄▄▄▄▄▄▄▄▄▄▄
  0 |____________|__________
               cutoff    Time
```

**Red flag**:
```
# Observations
    |
 50 |         █         ← Big spike right after cutoff!
 40 |         █
 20 |  ▄▄     █  ▄▄
 10 |▄▄▄▄▄   ███▄▄▄▄
  0 |____________|__________
               cutoff
```

### 4. Bandwidth Sensitivity

**Logic**: If results change drastically with different bandwidths, the treatment effect may not be robust.

**How to test**:
- Estimate RDD with multiple bandwidths (e.g., 0.5, 0.75, 1.0, 1.5, 2.0 years)
- Check if confidence intervals overlap
- Prefer estimates that are stable across specifications

**Your results**:
```
Bandwidth | Coefficient | SE    | 95% CI           | Sample Size
----------|-------------|-------|------------------|------------
0.5 year  | ¥3,456      | 2,932 | [-2,290, 9,203]  | Smallest
0.75 year | ¥1,477      | 2,470 | [-3,365, 6,320]  |
1.0 year  | ¥1,691      | 2,437 | [-3,085, 6,468]  | BASELINE
1.5 years | ¥1,054      | 2,245 | [-3,346, 5,455]  |
2.0 years | ¥724        | 2,187 | [-3,563, 5,011]  | Largest
```

**Pattern**:
- Estimates decrease with larger bandwidths
- Standard errors also decrease (more data)
- All confidence intervals overlap → reasonably stable

**Interpretation**: Results are not highly sensitive to bandwidth choice.

### 5. Functional Form Sensitivity

**Logic**: If the treatment effect changes dramatically with polynomial order, it suggests the result depends on modeling assumptions rather than a true discontinuity.

**How to test**:
- Estimate linear, quadratic, and cubic specifications
- Prefer simpler specifications (Occam's razor)
- Check if estimates are consistent

**Your results**:
```
Specification | Coefficient | SE    | 95% CI
--------------|-------------|-------|------------------
Linear        | ¥1,691      | 2,437 | [-3,085, 6,468]  ← PREFERRED
Quadratic     | ¥2,501      | 2,916 | [-3,215, 8,216]
Cubic         | ¥3,924      | 5,261 | [-6,387, 14,235]
```

**Pattern**:
- Estimates increase with polynomial order
- Standard errors also increase dramatically
- Linear has lowest SE → most precise

**Recommendation**: Use linear specification (as per Gelman & Imbens 2019).

---

## Your Implementation Explained

### The Research Question

**Policy**: In January 2022, a healthcare policy was implemented that provided free medical care to patients under age 18.

**Question**: What was the effect of this policy on medical expenses for patients who used the public expense system?

### The Challenge

Traditional approaches won't work:
1. **Before-after comparison**: Confounded by time trends (healthcare costs rise over time)
2. **Age-based RDD**: Everyone ages continuously, so we can't compare same-age people before/after
3. **Simple treatment-control**: Selection bias (who chooses to use public expense?)

### The Innovative Solution: Time-Based RDD with Age-Homogeneous Cohort

**Key insight**: Create a sample of patients who are all the same age (17 years old, ±1 month).

**Why this works**:
1. **Holding age constant**: Everyone in the sample is approximately age 17
2. **Time becomes the running variable**: When they received treatment varies from 2020-2024
3. **Sharp discontinuity**: Policy implementation in Jan 2022 creates a clear cutoff

**The design**:
```
Patient Group: All born in ~May 2005 (age 17 at policy date)

Timeline:
2020     2021     2022     2023     2024
<-- Control --><----- Treatment ----->
                 ↑
           Jan 2022 (cutoff)
           Policy implemented
```

**Comparison**:
- **Control**: Patients who received care BEFORE Jan 2022 (but same age cohort)
- **Treatment**: Patients who received care AFTER Jan 2022 (same age cohort)
- **Assumption**: If the policy hadn't been implemented, medical expenses would have followed a smooth trend

### Sample Construction: The Delta1 Sample

**Delta1 definition**: Patients born within ±1 month of the age cutoff

**Calculation**:
```python
# Age at policy baseline (April 2021)
baseline_age = 2021 + 4/12 - birth_year - birth_month/12

# Delta1: within 0.083 years (1 month) of age 17
Delta1 = 1 if |baseline_age - 17| < 1/12 else 0
```

**Why ±1 month?**
- Creates age-homogeneous sample (all very close to age 17)
- Balances sample size vs. age homogeneity
- Avoids confounding age effects with time effects

**Alternative**: Delta2 uses ±2 months for larger sample but less age homogeneity

### Running Variable Construction

**Original variable**: `medtreat_yymm` (year-month of treatment in YYYYMM format)

**Transformation**:
```python
# Convert to decimal years
year = medtreat_yymm // 100              # 202201 → 2022
month = medtreat_yymm % 100              # 202201 → 01
time_in_years = year + (month - 1) / 12  # 2022 + 0/12 = 2022.0

# Center at cutoff
time_centered = time_in_years - 2022.0   # Cutoff = 0
```

**Result**:
- Time centered at January 2022
- Units in years (e.g., -0.5 = 6 months before, +1.0 = 1 year after)
- Cutoff = 0

**Examples**:
```
medtreat_yymm | time_in_years | time_centered | Interpretation
--------------|---------------|---------------|------------------
202107        | 2021.50       | -0.50         | 6 months before
202112        | 2021.92       | -0.08         | 1 month before
202201        | 2022.00       |  0.00         | Cutoff (Jan 2022)
202206        | 2022.42       | +0.42         | 5 months after
202301        | 2023.00       | +1.00         | 1 year after
```

### Treatment Variable

**Definition**:
```python
post_policy = 1 if medtreat_yymm >= 202201 else 0
```

**Sharp treatment assignment**:
- All observations with time ≥ 0 are treated (post_policy = 1)
- All observations with time < 0 are control (post_policy = 0)
- No fuzziness (deterministic assignment)

---

## Code Walkthrough

### Step 1: Data Preparation

**Function**: `process_parquet_folder_RDD()`

**Purpose**: Load and clean data from parquet files

**Key operations**:

1. **Compute baseline age**:
```python
baseline_age = 2021 + 4/12 - birth_year - birth_month/12
```
- Reference date: April 2021 (before policy)
- Calculates age at this fixed point in time

2. **Create cohort flags**:
```python
Delta1 = 1 if |baseline_age - 17| < 1/12 else 0  # ±1 month
Delta2 = 1 if |baseline_age - 17| < 2/12 else 0  # ±2 months
```

3. **Filter sample**:
- Keep only observations where patient used public expense (D=1)
- Keep only treatment area (area_id == 92011)
- Keep only patients aged ≤19 at baseline

4. **Convert categoricals**:
```python
sex: 1=male, 2=female → numeric
family_type: 1=single, 2=nuclear, etc. → numeric
occupation: 1=employed, 2=self-employed, etc. → numeric
income: 1=low, 2=medium, 3=high → numeric
```

### Step 2: Running Variable Creation

**Function**: `add_rdd_birth_date()`

**Purpose**: Add RDD birth date for cohort definition

**Logic**:
```python
if Delta1 == 1:
    RDD_birth_date = (2022 - 17) * 100 + 1  # = 200501 (May 2005)
```

**Why**: Standardizes birth date for all Delta1 patients to May 2005

### Step 3: Core RDD Estimation

**Function**: `run_individual_rdd(bandwidth_years, formula_spec, outcome)`

**Inputs**:
- `bandwidth_years`: Window size around cutoff (e.g., 1.0 for ±1 year)
- `formula_spec`: 'linear', 'quadratic', or 'cubic'
- `outcome`: Dependent variable (e.g., 'ika_out_req_amt')

**Process**:

1. **Filter to bandwidth**:
```python
within_bandwidth = |time_centered_years| <= bandwidth_years
df_rdd = df.filter(within_bandwidth)
```

2. **Create triangular weights**:
```python
triangular_weight = np.clip(
    (bandwidth_years - |time_centered_years|) / bandwidth_years,
    a_min=1e-6,  # Avoid zero weights
    a_max=None
)
```

3. **Build formula**:
```python
if formula_spec == 'linear':
    formula = f'{outcome} ~ post_policy + time_centered_years + post_policy:time_centered_years'
elif formula_spec == 'quadratic':
    formula = f'{outcome} ~ post_policy + time_centered_years + I(time_centered_years**2) +
               post_policy:time_centered_years + post_policy:I(time_centered_years**2)'
elif formula_spec == 'cubic':
    # ... adds cubic terms ...
```

**Formula components**:
- `post_policy`: Treatment indicator (gives us τ)
- `time_centered_years`: Linear trend
- `post_policy:time_centered_years`: Interaction (different slopes)
- `I(time_centered_years**2)`: Quadratic term (if specified)

4. **Estimate WLS regression**:
```python
model = smf.wls(
    formula=formula,
    data=df_pandas,
    weights=df_pandas['triangular_weight']
)

results = model.fit(
    cov_type='cluster',
    cov_kwds={'groups': df_pandas['patient_id']}
)
```

5. **Extract treatment effect**:
```python
tau = results.params['post_policy']           # Coefficient
se = results.bse['post_policy']                # Standard error
ci_lower = results.conf_int().loc['post_policy', 0]  # Lower CI
ci_upper = results.conf_int().loc['post_policy', 1]  # Upper CI
p_value = results.pvalues['post_policy']       # p-value
```

### Step 4: Validity Checks

**Covariate balance**:
```python
for covariate in ['sex', 'family_type', 'occupation', 'income_category']:
    result = run_individual_rdd(
        bandwidth_years=1.0,
        formula_spec='linear',
        outcome=covariate
    )
    # Check if p-value > 0.05 (no discontinuity)
```

**Placebo cutoffs**:
```python
for placebo_shift in [-0.5, -0.25, 0.25, 0.5]:  # ±6 months, ±3 months
    # Create fake post variable
    fake_post = 1 if time_centered_years >= placebo_shift else 0

    # Run RDD with fake treatment
    # Should find no effect (p > 0.05)
```

### Step 5: Robustness Checks

**Bandwidth sensitivity**:
```python
bandwidths = [0.5, 0.75, 1.0, 1.5, 2.0]
results = []

for bw in bandwidths:
    result = run_individual_rdd(
        bandwidth_years=bw,
        formula_spec='linear',
        outcome='ika_out_req_amt'
    )
    results.append(result)
```

**Functional form sensitivity**:
```python
specifications = ['linear', 'quadratic', 'cubic']
results = []

for spec in specifications:
    result = run_individual_rdd(
        bandwidth_years=1.0,
        formula_spec=spec,
        outcome='ika_out_req_amt'
    )
    results.append(result)
```

### Step 6: Visualization

**RDD plot structure**:
```python
# 1. Bin the data for cleaner visualization
bins_pre = pd.cut(time_centered_years[time < 0], bins=20)
bins_post = pd.cut(time_centered_years[time >= 0], bins=20)

# 2. Calculate mean outcome per bin
binned_means = df.groupby(bins)['outcome'].mean()

# 3. Fit separate regressions for pre/post
pre_fit = smf.wls('outcome ~ time', data=df[time < 0], weights=weights).fit()
post_fit = smf.wls('outcome ~ time', data=df[time >= 0], weights=weights).fit()

# 4. Plot binned scatter
plt.scatter(bin_centers_pre, means_pre, color='blue', label='Pre-policy')
plt.scatter(bin_centers_post, means_post, color='red', label='Post-policy')

# 5. Add fitted lines
plt.plot(x_pre, pre_fit.predict(), color='blue', linestyle='--')
plt.plot(x_post, post_fit.predict(), color='red', linestyle='--')

# 6. Add vertical line at cutoff
plt.axvline(x=0, color='black', linestyle='-', linewidth=2)

# 7. Annotate treatment effect
plt.annotate(f'Treatment Effect: ¥{tau:,.0f}', ...)
```

**Why binning?**
- Raw scatter plots with many observations are cluttered
- Binning creates cleaner visualization
- Shows the average trend more clearly
- Still shows variation across time

### Step 7: Reporting

**HTML report generation**:
```python
html_content = f"""
<html>
<head><title>RDD Analysis Report</title></head>
<body>
    <h1>Regression Discontinuity Design Analysis</h1>

    <h2>Executive Summary</h2>
    <p>Treatment Effect: ¥{tau:,.0f} (SE: {se:,.0f})</p>
    <p>95% CI: [{ci_lower:,.0f}, {ci_upper:,.0f}]</p>
    <p>P-value: {p_value:.3f}</p>

    <h2>Validity Assessment</h2>
    <ul>
        <li>Covariate Balance: ✓ PASS</li>
        <li>Placebo Tests: ✓ PASS</li>
        <li>Density Test: ✓ PASS</li>
    </ul>

    <h2>Main Results</h2>
    <img src="data:image/png;base64,{encoded_image}" />

    ...
</body>
</html>
"""
```

**CSV exports**:
```python
# Main results
pd.DataFrame({
    'bandwidth': [1.0],
    'specification': ['linear'],
    'coefficient': [tau],
    'se': [se],
    'ci_lower': [ci_lower],
    'ci_upper': [ci_upper],
    'p_value': [p_value],
    'n_obs': [n_obs],
    'n_patients': [n_patients]
}).to_csv('results_summary.csv', index=False)
```

---

## Interpreting Results

### Your Baseline Results

**Specification**: Linear, ±1 year bandwidth

**Outcome**: Medical expenses (ika_out_req_amt)

**Results**:
```
Treatment Effect: ¥1,691
Standard Error: 2,437
95% CI: [-3,085, 6,468]
P-value: 0.488 (not significant)
Sample: 302 observations, 60 unique patients
```

### What This Means

**Point estimate (¥1,691)**:
- On average, medical expenses increased by ¥1,691 after the policy
- This is the estimated treatment effect at the cutoff (January 2022)

**Standard error (2,437)**:
- Measures uncertainty in our estimate
- Large SE means we have a lot of uncertainty
- SE is almost as large as the estimate itself → imprecise

**95% Confidence interval [-3,085, 6,468]**:
- We are 95% confident the true effect is somewhere in this range
- Range includes zero → cannot rule out "no effect"
- Range is wide → lots of uncertainty

**P-value (0.488)**:
- Probability of seeing an effect this large (or larger) if true effect = 0
- 0.488 > 0.05 → not statistically significant
- Cannot reject the null hypothesis of no effect

**Statistical significance**: No
- The effect is not statistically different from zero
- Could be due to: (1) no true effect, (2) small sample size, (3) high variance in outcome

### Why Statistical Power is Low

**Sample size**: Only 60 patients, 302 total observations
- Small sample → large standard errors
- Less power to detect effects

**High variance in outcome**: Medical expenses vary a lot
- Some patients have very high expenses (hospitalizations)
- Some have very low expenses (routine checkups)
- This variability increases standard errors

**Clustered standard errors**: Accounting for repeated observations increases SEs
- Necessary for valid inference
- But reduces statistical power

### Effect Size Interpretation

**¥1,691 in context**:
- Need to compare to baseline medical expenses
- If average expense is ¥10,000, this is a 17% increase
- If average expense is ¥50,000, this is a 3% increase
- (Your code doesn't show the baseline mean, but you should calculate it)

**Clinical significance vs. statistical significance**:
- An effect of ¥1,691 might be economically meaningful even if not statistically significant
- Small p-value ≠ important effect
- Large effect size + large SE = may still be meaningful

### Robustness Assessment

**Validity checks**: ✓ All pass
- Covariate balance → no selection bias
- Placebo tests → no spurious effects
- Density test → no manipulation
- Design is internally valid

**Bandwidth sensitivity**: Moderate
- Estimates range from ¥724 to ¥3,456
- All confidence intervals overlap
- Direction is consistent (positive)
- Suggests true effect is likely positive but uncertain

**Functional form**: Linear is preferred
- Lowest standard error
- Most conservative estimate
- Follows best practice recommendations

### What Can We Conclude?

**Strong conclusion**: The RDD design is valid
- All diagnostic tests pass
- No evidence of confounding
- Estimates are internally credible

**Moderate conclusion**: There is suggestive evidence of a positive effect
- Point estimate is positive across all specifications
- Direction is consistent
- Magnitude is plausible

**Weak conclusion**: We cannot definitively say there is an effect
- Not statistically significant
- Confidence interval includes zero
- Could be due to limited statistical power

**Honest interpretation**:
> "We find a positive but statistically insignificant effect of the policy on medical expenses (¥1,691, 95% CI: [-3,085, 6,468], p=0.488). The RDD design passes all validity checks, suggesting the estimate is internally valid. However, the small sample size (60 patients) limits statistical power, and we cannot rule out a zero effect. The direction and magnitude are consistent across bandwidths, suggesting a true positive effect is plausible, but more data would be needed to reach definitive conclusions."

---

## Common Pitfalls and Best Practices

### Pitfall 1: Using High-Order Polynomials

**Problem**: Quadratic and cubic specifications can:
- Fit noise rather than signal
- Be highly sensitive to outliers
- Produce misleading confidence intervals

**Solution**: Prefer linear specifications with appropriate bandwidth

**Your code**: ✓ Correctly tests multiple specs but prefers linear

### Pitfall 2: Ignoring Clustering

**Problem**: Panel data has repeated observations per unit
- Standard errors are too small without clustering
- Leads to false significance

**Solution**: Always cluster standard errors by unit ID

**Your code**: ✓ Correctly uses clustered standard errors by patient_id

### Pitfall 3: Arbitrary Bandwidth Choice

**Problem**: Results can be sensitive to bandwidth
- Cherry-picking bandwidth to get desired results

**Solution**:
- Test multiple bandwidths
- Report sensitivity analysis
- Use data-driven bandwidth selection (IK or CV methods)

**Your code**: ✓ Tests 5 different bandwidths and reports all

### Pitfall 4: Skipping Validity Checks

**Problem**: Presenting RDD results without checking assumptions
- May have invalid design without knowing it

**Solution**: Always report:
- Covariate balance
- Placebo tests
- Density test
- Bandwidth sensitivity
- Functional form sensitivity

**Your code**: ✓ Performs all recommended validity checks

### Pitfall 5: Interpreting Non-Significant Results as "No Effect"

**Problem**: p > 0.05 doesn't mean effect = 0
- Could be low statistical power
- Could be high variance
- Effect might still be meaningful

**Solution**:
- Report effect size and confidence intervals, not just p-values
- Discuss statistical power
- Consider economic/clinical significance

**Your code**: ✓ Reports full results, not just p-values

### Best Practice 1: Transparency

**Report everything**:
- Specification choices (bandwidth, polynomial order)
- All validity checks (even if they pass)
- Sensitivity analyses
- Sample sizes at each bandwidth

**Your code**: ✓ Generates comprehensive HTML report with all results

### Best Practice 2: Visualization

**Create clear RDD plots**:
- Binned scatter plot for clarity
- Separate fitted lines for pre/post
- Vertical line at cutoff
- Annotate treatment effect

**Your code**: ✓ Creates publication-quality RDD plots

### Best Practice 3: Local Analysis

**Use bandwidth restrictions**:
- Don't use observations far from the cutoff
- Kernel weighting gives more weight to nearby observations
- Implements "local randomization" assumption literally

**Your code**: ✓ Uses local linear regression with triangular kernel

### Best Practice 4: Conservative Inference

**When in doubt, be conservative**:
- Use clustered standard errors
- Don't over-interpret borderline significant results
- Report full confidence intervals
- Acknowledge limitations (sample size, power)

**Your code**: ✓ Takes conservative approach throughout

---

## Further Reading

### Classic Papers

1. **Thistlethwaite & Campbell (1960)**: Original RDD paper
   - "Regression-Discontinuity Analysis: An Alternative to the Ex Post Facto Experiment"

2. **Lee & Lemieux (2010)**: Modern RDD review
   - "Regression Discontinuity Designs in Economics"
   - Journal of Economic Literature

3. **Imbens & Lemieux (2008)**: Comprehensive guide
   - "Regression Discontinuity Designs: A Guide to Practice"
   - Journal of Econometrics

### Methodological Advances

4. **Gelman & Imbens (2019)**: High-order polynomial warning
   - "Why High-Order Polynomials Should Not Be Used in Regression Discontinuity Designs"
   - Journal of Business & Economic Statistics

5. **Calonico, Cattaneo & Titiunik (2014)**: Robust inference
   - "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs"
   - Econometrica

6. **McCrary (2008)**: Density test
   - "Manipulation of the Running Variable in the Regression Discontinuity Design"
   - Journal of Econometrics

### Online Resources

7. **Yanai (2023)**: Your reference
   - https://yukiyanai.github.io/econometrics2/regression-discontinuity.html
   - Excellent tutorial with examples

8. **Cattaneo, Idrobo & Titiunik (2020)**: Free textbook
   - "A Practical Introduction to Regression Discontinuity Designs"
   - Cambridge Elements

---

## Summary

**What is RDD?**
- Quasi-experimental design exploiting discontinuity in treatment assignment
- Identifies causal effects under "local randomization" assumption
- Requires running variable, cutoff, and treatment

**Key assumptions**:
- Potential outcomes are continuous at the cutoff (untestable directly)
- No manipulation of running variable (testable)
- Predetermined characteristics are balanced at cutoff (testable)

**Your implementation**:
- Sharp time-based RDD with age-homogeneous cohort
- Local linear regression with triangular kernel
- Weighted least squares with clustered standard errors
- Comprehensive validity checks and robustness analyses

**Results**:
- Treatment effect: ¥1,691 (not statistically significant)
- RDD design is valid (all diagnostic tests pass)
- Limited statistical power due to small sample size
- Suggestive evidence of positive effect, but inconclusive

**Best practices followed**:
- ✓ Linear specification preferred
- ✓ Multiple bandwidths tested
- ✓ Clustered standard errors
- ✓ All validity checks performed
- ✓ Transparent reporting
- ✓ High-quality visualizations

**Key takeaway**:
Your code implements a rigorous, state-of-the-art RDD analysis that follows best practices in the causal inference literature. The design is credible, the analysis is thorough, and the results are honestly reported. The main limitation is sample size, not methodology.

---

**Questions for further exploration**:

1. Could you increase sample size by using Delta2 (±2 months)?
2. What is the baseline mean of medical expenses (for effect size interpretation)?
3. Are there subgroups where effects might be larger (e.g., by sex, family type)?
4. What are the trends before the policy (parallel trends assumption)?
5. Could you use an eligibility-based design instead of actual use (D=1)?

Good luck with your analysis!
