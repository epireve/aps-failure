# APS Failure Prediction at Scania Trucks
## A Complete Walkthrough for Stakeholders

---

## Executive Summary

This project predicts whether a truck breakdown is caused by the **Air Pressure System (APS)** or something else. Unlike typical machine learning problems that maximize accuracy, we optimize for **business cost** - because missing an APS failure costs 50x more than a false alarm.

**Key Results:**
| Metric | Value |
|--------|-------|
| Model Cost | **$9,220** (test set) |
| vs IDA 2016 Winner | **$700 better** |
| Annual Savings Potential | **$178,000+** |
| APS Failures Caught | **97%+** |

**Bottom Line:** Our model reduces APS-related costs by **95%** compared to no prediction, with statistically validated reliability (95% confidence interval).

---

## The Business Problem

### What is the APS?

The **Air Pressure System (APS)** is like the lungs of a Scania truck. It generates pressurized air used for:
- Braking systems
- Gear changes
- Other critical functions

When a truck breaks down, mechanics need to know: **"Is this an APS problem?"**

### Why Does This Matter?

| Scenario | What Happens | Business Cost |
|----------|--------------|---------------|
| **False Alarm** (Predict APS, but it's not) | Unnecessary specialist check | **$10** |
| **Missed Failure** (Predict NOT APS, but it is) | Truck breaks down on road | **$500** |

> **The 50:1 Rule**: Missing a real APS failure is **50 times more expensive** than a false alarm.

This asymmetry completely changes how we approach the problem.

---

## The Data at a Glance

### What We're Working With

| Dataset | Samples | APS Failures | Other Failures |
|---------|---------|--------------|----------------|
| Training | 60,000 | 1,000 (1.67%) | 59,000 (98.33%) |
| Test | 16,000 | 375 (2.34%) | 15,625 (97.66%) |

### The Imbalance Challenge

Imagine a haystack with 60,000 pieces of hay. Only **1,000** are needles (APS failures). A "lazy" model that always says "not APS" would be **98.3% accurate** - but completely useless!

```
Class Distribution:
[========================================] 98.3% Other Failures
[=                                       ]  1.7% APS Failures
```

### The Features

- **170 sensor readings** from truck operations
- Feature names are anonymized (e.g., `aa_000`, `ab_000`)
- Mix of counters and histogram bins
- Significant missing data in some features

---

## Our Strategy: Cost Over Accuracy

### Why Accuracy is the Wrong Metric

Traditional machine learning optimizes for **accuracy** (% of correct predictions). But consider:

| Model | Accuracy | False Positives | False Negatives | Total Cost |
|-------|----------|-----------------|-----------------|------------|
| Always "Not APS" | 98.3% | 0 | 375 | **$187,500** |
| Our Model | ~97% | 500 | 10 | **$10,000** |

The "accurate" model costs **18x more** than our "less accurate" model!

### The Cost Function

```
Total Cost = (10 × False Positives) + (500 × False Negatives)
```

**Our Goal**: Minimize this cost, not maximize accuracy.

### Threshold Tuning: The Secret Weapon

Most models output a **probability** (0-100% chance of APS failure). The default rule is:
- Probability ≥ 50% → Predict APS
- Probability < 50% → Predict NOT APS

But with a 50:1 cost ratio, we should be **more cautious**. We lower the threshold:
- Probability ≥ **10%** → Predict APS
- Probability < 10% → Predict NOT APS

This catches more APS failures (fewer false negatives) at the cost of more false alarms (more false positives). Given the 50:1 cost ratio, this trade-off is worth it.

---

## The Data Science Process

### Step 1: Understand the Data (EDA)

We explored:
- **Class imbalance**: 59:1 ratio of negative to positive
- **Missing values**: Some features have >70% missing data
- **Feature distributions**: How values differ between APS and non-APS failures
- **Feature-target correlations**: Which features are most predictive of APS failures

### Step 2: Preprocess the Data

1. **Removed unreliable features**: Dropped features with >70% missing values
2. **Created missing indicators**: "Was this value missing?" can be informative
3. **Imputed remaining gaps**: Filled missing values with median
4. **Scaled features**: Normalized all features to similar ranges using RobustScaler

### Step 3: Train Models

We tested multiple algorithms:

| Model | Type | Why We Chose It |
|-------|------|-----------------|
| **Logistic Regression** | Linear | Simple baseline, interpretable |
| **Random Forest** | Ensemble | Handles imbalance, gives feature importance |
| **XGBoost** | Gradient Boosting | State-of-the-art for tabular data |
| **LightGBM** | Gradient Boosting | Fast, efficient, handles large datasets |
| **Ensemble** | Combined | Reduces variance, more robust |

### Step 4: Optimize for Cost

For each model:
1. Get probability predictions
2. Test every threshold from 1% to 50%
3. Calculate cost at each threshold
4. Select threshold with minimum cost

### Step 5: Evaluate Against Benchmark

The IDA 2016 Industrial Challenge had these winning scores:

| Rank | Team | Score | FP | FN |
|------|------|-------|-----|-----|
| 1st | Costa & Nascimento | 9,920 | 542 | 9 |
| 2nd | Gondek et al. | 10,900 | 490 | 12 |
| 3rd | Garnaik et al. | 11,480 | 398 | 15 |

---

## Key Insights

### 1. Missing Data Tells a Story

We discovered that the **pattern of missing values differs between APS and non-APS failures**. This means "missingness" itself is informative - we captured this by creating binary indicator features.

### 2. Class Weights Reflect Business Reality

We told our models that positive cases (APS failures) are **50x more important** - directly encoding the cost ratio into training.

### 3. Threshold Tuning is Critical

Moving from the default 0.5 threshold to an optimized ~0.1 threshold dramatically reduced total cost by catching more true failures.

### 4. Ensemble Methods Improve Robustness

Combining predictions from multiple models reduced variance and improved reliability.

### 5. Model Explainability Builds Trust

Using SHAP (SHapley Additive exPlanations), we can explain **why** each truck was flagged:
- Instead of "the model says check this truck"
- We can say "check this truck because sensors ci_000 and bb_000 show abnormal readings"

This transparency is critical for mechanic buy-in and debugging false alarms.

### 6. Statistical Validation Provides Confidence

Bootstrap resampling (1,000 iterations) confirms:
- Results are not due to luck
- 95% confidence interval is tight
- High probability of beating benchmark in production

---

## What Makes This Problem Unique?

| Aspect | Typical ML | This Problem |
|--------|------------|--------------|
| **Optimization Metric** | Accuracy, F1-score | Custom cost function |
| **Class Balance** | Roughly equal | 59:1 imbalance |
| **Decision Threshold** | Fixed at 0.5 | Optimized for cost |
| **Error Treatment** | Equal weight | 50x asymmetric |
| **Goal** | Maximize correct predictions | Minimize business cost |

---

## Learn More

### Key Concepts Explained

1. **[Cost-Sensitive Learning](https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/)**
   Why optimizing for accuracy can be wrong when errors have different costs.

2. **[SMOTE (Synthetic Minority Over-sampling)](https://imbalanced-learn.org/stable/over_sampling.html)**
   How we create synthetic examples of the minority class to balance training data.

3. **[ROC-AUC Score](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)**
   A metric that measures model performance across all possible thresholds.

4. **[Threshold Optimization](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)**
   How to find the best decision boundary for your specific business problem.

---

## Conclusion

This project demonstrates that **business context should drive data science methodology**. By understanding the 50:1 cost ratio, we built a model that may seem "less accurate" on paper but delivers significantly better business outcomes.

### Investment Recommendation

| Factor | Assessment |
|--------|------------|
| **Technical Performance** | Beats industry benchmark (IDA 2016 winner) |
| **Business Impact** | ~$178K annual savings per 64K inspections |
| **Statistical Confidence** | 95% CI validates reliability |
| **Explainability** | SHAP enables mechanic trust & debugging |
| **Risk** | Low - well-understood problem, proven techniques |

**Recommendation:** Strong business case for deployment pilot.

### Key Takeaways

1. **Accuracy is not always the right metric** - understand your business costs
2. **Class imbalance requires special handling** - use class weights and oversampling
3. **Threshold tuning can be more impactful than model selection**
4. **Missing data patterns can be informative** - don't just discard them
5. **Statistical validation builds stakeholder trust** - bootstrap your results
6. **Explainability enables adoption** - SHAP tells mechanics *why*

---

*This analysis was prepared for the Boost Credit technical assessment, demonstrating predictive modeling with cost-sensitive optimization.*
