# DSN Bootcamp Qualification Hackathon 2026: ML Track
**Author:** Sufyan Zakariyya Sani  
**Challenge:** Predict total sales for a given product at a given store format (Regression Task evaluated via Root Mean Squared.

---

## Project Overview
This repository contains the end-to-end machine learning pipeline developed for the Data Science Nigeria (DSN) AI Bootcamp Qualification Hackathon[cite: 1]. The objective is to analyze retail store sales data across Nigeria and build a robust predictive model to optimize inventory, pricing, and retail strategies[cite: 1].

## Solution Architecture & Methodology

### 1. Exploratory Data Analysis & Data Cleaning
- **Text Standardization:** Normalized string casing across categorical columns to prevent duplicate label fragmentation.
- **Intelligent Imputations:** 
  - Missing `product_weight_kg` values were imputed using the group mean of the corresponding `product_code`.
  - Missing `store_size` values were mapped from matching `store_code` outlets.
  - Corrected physical anomalies where `shelf_visibility` was recorded as `0.0`, treating them as missing values and replacing them with product-level means.

### 2. Advanced Feature Engineering
- **Price per KG:** Engineered a density-pricing feature (`product_price / product_weight_kg`).
- **Category Price Ratio:** Compared individual product prices against their respective category averages to capture premium vs. budget positioning.
- **Store Traffic Proxy:** Mapped total product listings per store (`store_product_count`) to account for store format scale.
- **Out-of-Fold (OOF) Target Encoding:** Applied smoothed historical sales target encoding to high-cardinality features (`product_code`, `store_code`) using a strict 5-fold cross-validation loop to prevent data leakage.

### 3. Modeling & Blended Ensemble
Instead of relying on a single estimator, the final submission utilizes a weighted **Blended Ensemble**:
- **HistGradientBoostingRegressor (60% weight):** Configured with L2 regularization and shallow tree depth to prevent overfitting.
- **GradientBoostingRegressor (40% weight):** Tuned sequentially to capture residual non-linear patterns.
- **Validation Strategy:** Evaluated rigorously using 5-Fold Cross-Validation, minimizing local Root Mean Squared Error (RMSE).

---

## Repository Structure
- `notebook/dsn_bootcamp_solution.ipynb`: The complete step-by-step Jupyter Notebook containing EDA, feature engineering, and training.
- `outputs/dsn_bootcamp_final_submission.csv`: The final optimized prediction file for Kaggle submission.
- `requirements.txt`: Required python packages.
