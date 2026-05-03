# Olist Brazilian E-Commerce — Churn Prediction Project Report

**Course:** Machine Learning / Data Science  
**Dataset:** Olist Brazilian E-Commerce Public Dataset (Kaggle)  
**Date:** May 2026

---

## 1. Executive Summary

This project applies the full data science lifecycle — loading, cleaning, feature engineering, RFM segmentation, preprocessing, and supervised machine learning — to predict customer churn in the Olist Brazilian e-commerce marketplace. The final best model is a **Tuned XGBoost classifier** achieving an AUC of **0.8267**, Accuracy of **82.67%**, F1-Score of **0.8194**, and Cohen's Kappa of **0.6533** on a balanced 50/50 test set.

The work is organized into four role-specific Jupyter notebooks and deployed as both an interactive React dashboard and a Streamlit web application.

---

## 2. Dataset Description

### 2.1 Source
Olist Brazilian E-Commerce Public Dataset — available on Kaggle. Contains anonymised transactional data from the Olist Store marketplace covering 2016–2018.

### 2.2 Files Used

| File | Description |
|------|-------------|
| `olist_orders_dataset.csv` | Core order-level data (status, timestamps) |
| `olist_order_items_dataset.csv` | Per-item price, freight, seller info |
| `olist_order_payments_dataset.csv` | Payment type, installments, value |
| `olist_order_reviews_dataset.csv` | Customer review scores and comments |
| `olist_customers_dataset.csv` | Customer ID and geolocation |
| `olist_products_dataset.csv` | Product category and dimensions |
| `olist_sellers_dataset.csv` | Seller geolocation |
| `olist_geolocation_dataset.csv` | ZIP-code lat/lng mapping |
| `product_category_name_translation.csv` | Portuguese → English category names |

### 2.3 Merge Result
After joining all tables on their respective keys and filtering to delivered orders only: **95,104 rows × 20+ columns**.

---

## 3. Exploratory Data Analysis

Key findings during EDA:
- **Delivery performance:** Median delivery time ≈ 12 days; ~7% of orders delivered late.
- **Review scores:** Mean = 4.09/5; heavily left-skewed (most scores = 5).
- **Payment:** Credit card dominates (~74%); average 2.9 installments.
- **Geolocation:** São Paulo state accounts for ~42% of orders.
- **Missing values:** `order_approved_at`, `order_delivered_carrier_date`, and `order_delivered_customer_date` contain nulls — handled via imputation or row removal.

---

## 4. Feature Engineering & RFM Analysis

### 4.1 RFM Definition

| Metric | Definition | Formula |
|--------|-----------|---------|
| **Recency** | Days since last purchase | `max_date − last_order_date` |
| **Frequency** | Number of distinct orders | `count(order_id)` per `customer_unique_id` |
| **Monetary** | Total spend | `sum(payment_value)` per `customer_unique_id` |

### 4.2 RFM Segments

| Segment | Count | Share |
|---------|-------|-------|
| Champions | 412 | 4.56% |
| Loyal Customers | 683 | 7.56% |
| Potential Loyalists | 1,024 | 11.34% |
| At Risk | 2,187 | 24.21% |
| Lost Customers | 4,732 | 52.33% |

> **Insight:** 76.5% of customers are either *At Risk* or *Lost*, confirming severe one-time-buyer behaviour in the platform.

### 4.3 Churn Definition

**Churn = Frequency of 1** (customer placed exactly one order and never returned).  
- Original churn rate: **97%** of unique customers.
- This extreme imbalance motivated the 50/50 balancing strategy.

### 4.4 Features Selected (13)

After dropping `no_of_orders` to prevent target leakage:

`payment_sequential`, `payment_installments`, `payment_value`, `review_score`, `order_item_id`, `no_of_products`, `price`, `freight_value`, `purchased_approved`, `approved_carrier`, `carrier_delivered`, `delivered_estimated`, `purchased_delivered`

---

## 5. Preprocessing

### 5.1 Steps Applied

1. **IQR capping** — Outliers beyond 1.5 × IQR replaced with fence values for all numeric features.
2. **PowerTransformer (Yeo-Johnson)** — Applied to correct skewness and approach normality.
3. **Target leakage removal** — `no_of_orders` dropped (directly encodes frequency/churn).

### 5.2 Result
After preprocessing and dropping rows with missing critical fields: **9,038 rows × 13 features**.

### 5.3 Balancing Strategy

To address the 97% churn imbalance:
- Random under-sampling of the majority class (churned) to match the minority class (retained).
- Final balanced dataset: **374 churned + 374 retained = 748 rows**.
- Train/test split: **80/20 → 598 train, 150 test** (stratified).

---

## 6. Model Development

### 6.1 Baseline Models (7 algorithms)

All trained on the balanced 598-sample training set, evaluated on 150-sample test set.

| Model | AUC | Accuracy | F1 | Kappa |
|-------|-----|----------|----|-------|
| XGBoost | 0.8000 | 0.8000 | 0.7972 | 0.6000 |
| Random Forest | 0.7800 | 0.7800 | 0.7762 | 0.5600 |
| Logistic Regression | 0.7800 | 0.7800 | 0.7762 | 0.5600 |
| Naïve Bayes | 0.7600 | 0.7600 | 0.7547 | 0.5200 |
| SVM | 0.7400 | 0.7400 | 0.7333 | 0.4800 |
| KNN (k=3) | 0.7333 | 0.7333 | 0.7278 | 0.4667 |
| Decision Tree | 0.7067 | 0.7067 | 0.6994 | 0.4133 |

> **Note:** Logistic Regression and KNN k=2 were excluded from the final comparison per project requirements. KNN used k=3. SVM was added.

### 6.2 Recursive Feature Elimination (RFE)

Applied RFE with 4 estimators (Random Forest, XGBoost, Logistic Regression, SVM) to identify the top **6 features**:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | freight_value | 0.182 |
| 2 | carrier_delivered | 0.161 |
| 3 | approved_carrier | 0.129 |
| 4 | order_item_id | 0.089 |
| 5 | no_of_products | 0.062 |
| 6 | payment_sequential | 0.058 |

**XGBoost (RFE)** achieved AUC = 0.7933 using only these 6 features — competitive with the 13-feature baseline.

### 6.3 GridSearchCV Hyperparameter Tuning

Three models tuned: Random Forest, XGBoost, SVM.

**XGBoost search space:**
```
learning_rate: [0.01, 0.05, 0.1, 0.2]
max_depth:     [3, 5, 7]
n_estimators:  [100, 200, 300]
```

**Best parameters found:**
- `learning_rate = 0.1`
- `max_depth = 3`  
- `n_estimators = 200`

### 6.4 Final Model Comparison (All 12 Models)

| Rank | Model | Group | AUC | Accuracy | F1 | Kappa |
|------|-------|-------|-----|----------|----|-------|
| **1** | **Tuned XGBoost** | **Tuned** | **0.8267** | **82.67%** | **0.8194** | **0.6533** |
| 2 | XGBoost | Baseline | 0.8000 | 80.00% | 0.7972 | 0.6000 |
| 3 | XGBoost (RFE) | RFE | 0.7933 | 79.33% | 0.7870 | 0.5867 |
| 4 | Random Forest | Baseline | 0.7800 | 78.00% | 0.7762 | 0.5600 |
| 5 | Logistic Regression | Baseline | 0.7800 | 78.00% | 0.7762 | 0.5600 |
| 6 | Naïve Bayes | Baseline | 0.7600 | 76.00% | 0.7547 | 0.5200 |
| 7 | Tuned SVM | Tuned | 0.7533 | 75.33% | 0.7481 | 0.5067 |
| 8 | Tuned Random Forest | Tuned | 0.7533 | 75.33% | 0.7481 | 0.5067 |
| 9 | SVM (RFE) | RFE | 0.7467 | 74.67% | 0.7414 | 0.4933 |
| 10 | SVM | Baseline | 0.7400 | 74.00% | 0.7333 | 0.4800 |
| 11 | KNN (k=3) | Baseline | 0.7333 | 73.33% | 0.7278 | 0.4667 |
| 12 | Decision Tree | Baseline | 0.7067 | 70.67% | 0.6994 | 0.4133 |

---

## 7. Best Model Results — Tuned XGBoost

### 7.1 Confusion Matrix (Test Set, 150 samples)

```
                  Predicted: Retained   Predicted: Churned
Actual: Retained        65                   10
Actual: Churned         16                   59
```

### 7.2 Derived Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 82.67% |
| Precision (Churn) | 85.5% |
| Recall / Sensitivity (Churn) | 78.7% |
| F1-Score | 0.8194 |
| Specificity (Retained) | 86.7% |
| Cohen's Kappa | 0.6533 |
| AUC-ROC | 0.8267 |

### 7.3 Interpretation

- The model correctly identifies **78.7%** of churned customers (59 out of 75).
- When it predicts churn, it is correct **85.5%** of the time.
- **16 false negatives** (missed churners) represent the highest-risk business case — customers who left but were not flagged.
- **10 false positives** (unnecessary retention spend) — a manageable cost.

---

## 8. Notebooks Structure

The original monolithic 553-cell Kaggle notebook was split into 4 role-specific notebooks:

| Notebook | Role | Responsibility |
|----------|------|----------------|
| `01_Data_Loading_and_EDA.ipynb` | Data Engineer / Analyst | Load 9 CSVs, merge, EDA, basic visualisations |
| `02_Feature_Engineering_RFM.ipynb` | Data Scientist | RFM computation, churn labelling, initial feature selection |
| `03_Preprocessing_Balancing.ipynb` | ML Engineer | Outlier capping, PowerTransformer, balancing (50/50), train/test split |
| `04_Model_Training_Evaluation.ipynb` | ML Engineer / Lead | All 12 models: baseline → RFE → GridSearchCV → evaluation |

---

## 9. Deployment

### 9.1 React Dashboard (`/churn-dashboard/`)

A production-quality interactive dashboard built with React + Vite + Recharts, served through a Node.js/Express API server using a contract-first OpenAPI design.

**Components:**
- 4 KPI cards
- Model AUC horizontal bar chart (all 12 models)
- RFM customer segments pie chart
- Feature importance horizontal bar chart (RFE vs. non-RFE highlighted)
- Confusion matrix heatmap with Precision / Recall / Accuracy callouts
- Multi-metric radar chart (top 5 models)
- Best model detail card (params + metrics)
- Full sortable model comparison table
- Pipeline summary cards

**Features:** Dark mode toggle, PDF export, CSV export per chart, split refresh with auto-refresh dropdown.

**API:** 5 REST endpoints served by Express (`/api/churn/*`) with Zod validation and React Query hooks auto-generated from the OpenAPI spec.

### 9.2 Streamlit App (`deployment/app.py`)

A Python Streamlit app providing the same analytical content in a data-science-friendly format, suitable for rapid sharing with technical stakeholders.

**Tabs:**
1. Model Comparison — AUC bar chart, radar chart, full results table
2. Feature Importance — importance chart with RFE highlights, side-by-side tables
3. Confusion Matrix — heatmap + derived metrics + business interpretation
4. RFM Segments — pie + bar + segment table + key finding alert
5. Pipeline Summary — accordion step-by-step + at-a-glance metrics + notebook roles

**Run command:**
```bash
cd deployment
streamlit run app.py --server.port 5000
```

---

## 10. Key Findings & Business Recommendations

### 10.1 Key Findings

1. **Extreme one-time-buyer behaviour:** 97% of Olist customers never reorder. This is a structural platform challenge, not a model artefact.
2. **Freight cost is the top predictor:** `freight_value` (importance 0.182) is the single strongest signal. High freight costs correlate strongly with non-return.
3. **Delivery chain matters:** `carrier_delivered` and `approved_carrier` (both RFE-selected) indicate that logistics performance is the second strongest churn driver.
4. **Review score contributes but ranks 7th:** Customer satisfaction, while important, is less predictive than delivery economics.
5. **XGBoost consistently outperforms:** Across all configurations (baseline, RFE, tuned), XGBoost leads. Tuning `max_depth=3` prevents overfitting on the small balanced dataset.

### 10.2 Business Recommendations

| Priority | Recommendation | Rationale |
|----------|---------------|-----------|
| **High** | Reduce freight costs or offer free shipping thresholds | Top predictor; directly actionable |
| **High** | Improve carrier SLA monitoring | `carrier_delivered` and `approved_carrier` are top-6 RFE features |
| **Medium** | Targeted retention campaigns for "At Risk" segment (24%) | High-volume group, still recoverable |
| **Medium** | Post-delivery follow-up for customers with review score ≤ 3 | Review score ranks 7th — still informative |
| **Low** | Incentivise second purchases within 30 days | Frequency-1 definition means any repeat = retained |

---

## 11. Limitations & Future Work

- **Small balanced dataset (748 rows):** The aggressive under-sampling discards 97% of churned records. Future work should use SMOTE or cost-sensitive learning on the full 9,038 rows.
- **Static churn definition:** Frequency = 1 is binary. A time-windowed churn definition (e.g., no purchase in 180 days) would be more realistic.
- **No temporal validation:** Models were validated on a random split. Time-based validation (train on 2016–2017, test on 2018) would better reflect deployment risk.
- **Feature interactions not modelled explicitly:** XGBoost captures them implicitly, but SHAP analysis could reveal interaction patterns.
- **No deployment pipeline:** A production ML pipeline (feature store, model registry, scheduled retraining) was not built in this phase.

---

## 12. Technical Stack Summary

| Component | Technology |
|-----------|-----------|
| Language (ML) | Python 3.11 |
| ML Libraries | scikit-learn, XGBoost |
| Data Processing | pandas, numpy |
| Visualisation (Python) | plotly, matplotlib, seaborn |
| Dashboard (React) | React 18 + Vite 7 + Recharts + TanStack Query |
| API Server | Express 5 + Zod + Orval codegen |
| Streamlit App | Streamlit 1.32 + Plotly |
| Monorepo | pnpm workspaces |
| Deployment | Replit (Node.js + Python) |

---

*End of Report*
