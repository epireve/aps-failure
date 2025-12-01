# APS Failure Prediction at Scania Trucks

Predictive model for Air Pressure System (APS) failures in heavy Scania trucks using cost-sensitive machine learning. For full analysis and code, see [Notebook](aps_failure_analysis.ipynb).

## Business Context
Missing an APS failure costs **$500** vs **$10** for a false alarm - a 50:1 cost ratio. This asymmetric cost (official from Scania CV AB) drives our entire optimization strategy: we minimize business cost, not accuracy.

## Data Source
**[UCI ML Repository - APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)**

| Detail | Value |
|--------|-------|
| Provider | Scania CV AB, Sweden |
| Challenge | IDA 2016 Industrial Challenge |
| Training | 60,000 samples (1.67% positive) |
| Features | 170 anonymized sensor readings |

## Approach
1. Cost-sensitive preprocessing with missing value indicators
2. Gradient boosting models (XGBoost, LightGBM) with class weights
3. Threshold optimization to minimize total cost
4. Ensemble averaging for robustnes

## Results
| Model | Cost | vs IDA Winner |
|-------|------|---------------|
| **Our Ensemble** | $9,220 | -$700 (better) |
| IDA 2016 1st Place | $9,920 | baseline |
