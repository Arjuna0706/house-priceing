# 🏠 King County House Price Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **A complete end-to-end data science project** — from raw data with missing values to a fully optimized Ridge Regression model — predicting house sale prices in King County, Washington (USA).

---

## 📌 Project Overview

This project explores **21,613 real estate transactions** in King County (which includes Seattle) to understand what drives house prices and build a reliable predictive model.

The analysis goes beyond simple regression — it handles real-world data challenges like missing values, evaluates multiple modeling strategies, and applies **Polynomial Feature Engineering** with **Ridge Regularization** to maximize prediction accuracy.

---

## 🎯 Business Problem

> *Can we accurately predict the sale price of a house given its physical attributes and location?*

Real estate pricing is notoriously complex. This project tackles that problem by identifying the most influential features — from square footage to waterfront access — and engineering a model that generalizes well to unseen data.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | King County House Sales (Seattle, WA) |
| **Records** | 21,613 transactions |
| **Features** | 21 variables |
| **Target** | `price` (USD) |
| **Price Range** | $75,000 — $7,700,000 |
| **Average Price** | ~$540,088 |

### Key Features

| Feature | Description |
|---|---|
| `sqft_living` | Interior square footage of the home |
| `grade` | Overall grade given to the housing unit |
| `sqft_above` | Square footage above ground level |
| `bathrooms` | Number of bathrooms |
| `view` | Quality of the view from the property |
| `waterfront` | Whether the property has waterfront access |
| `lat` / `long` | Geographic coordinates |
| `yr_built` | Year the house was built |
| `yr_renovated` | Year of last renovation |

---

## 🔍 Key Insights

### 1. Most Influential Features on Price

```
sqft_living      → 0.702   ████████████████████  Strongest predictor
grade            → 0.667   ████████████████████
sqft_above       → 0.606   ██████████████████
sqft_living15    → 0.585   █████████████████
bathrooms        → 0.525   ████████████████
view             → 0.397   ████████████
sqft_basement    → 0.321   █████████
lat              → 0.307   █████████
waterfront       → 0.266   ████████
```

### 2. Waterfront Premium
Properties with **waterfront access** command significantly higher prices than non-waterfront properties — a clear outlier driver that the boxplot analysis confirms.

### 3. Square Footage is King
The strongest linear relationship with price is `sqft_living` (r = 0.702). Larger homes consistently sell for more, with a clear upward trend confirmed via regression plot.

### 4. Location Matters
Latitude has a moderate positive correlation with price (r = 0.307), indicating that properties in the northern areas of King County tend to be priced higher.

---

## 🧪 Methodology

```
Raw Data  →  Data Cleaning  →  EDA  →  Feature Selection  →  Modeling  →  Evaluation
```

### Step 1: Data Wrangling
- Dropped irrelevant columns (`id`, `Unnamed: 0`)
- Identified and handled **13 missing values** in `bedrooms` and **10** in `bathrooms`
- Imputed missing values using **column mean** strategy

### Step 2: Exploratory Data Analysis
- Correlation heatmap across all numerical features
- Boxplot analysis for categorical vs. price (waterfront)
- Regression plot for continuous feature vs. price (sqft_above)

### Step 3: Modeling Pipeline

| Model | Approach | R² Score |
|---|---|---|
| Simple Linear Regression (longitude) | Baseline | ~0.000 |
| Simple Linear Regression (sqft_living) | Single feature | ~0.493 |
| Multiple Linear Regression | All features | ~0.658 |
| Polynomial + Pipeline | Feature engineering | ~0.751 |
| **Ridge Regression (Polynomial)** | **Best model** | **~0.798** |

### Step 4: Train-Test Split
- **85% training** / **15% testing** with `random_state=1`
- 18,371 training samples / 3,242 test samples

---

## 🛠️ Tech Stack

```python
pandas       # Data manipulation and cleaning
numpy        # Numerical computing
matplotlib   # Data visualization
seaborn      # Statistical plotting
scikit-learn # Machine learning pipeline
  ├── LinearRegression
  ├── Ridge
  ├── PolynomialFeatures
  ├── StandardScaler
  ├── Pipeline
  └── cross_val_score
```

---

## 📁 Project Structure

```
📦 house-price-prediction/
 ┣ 📓 House_Priceing.ipynb    ← Main analysis notebook
 ┣ 📊 kc_house_data_NaN.csv   ← Raw dataset with missing values
 ┗ 📄 README.md               ← You are here
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
jupyter notebook House_Priceing.ipynb
```

---

## 📈 Results Summary

The final model — **Ridge Regression with Polynomial Features (degree=2)** — achieves an **R² score of ~0.798** on the test set, meaning it explains approximately **79.8% of the variance** in house prices.

| Metric | Value |
|---|---|
| Best Model | Ridge Regression + Polynomial Features |
| R² Score (Test) | ~0.798 |
| Alpha (Ridge) | 0.1 |
| Polynomial Degree | 2 |

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` for optimal Ridge alpha
- [ ] Feature engineering: price per sqft, age of house, renovation flag
- [ ] Geospatial visualization using Folium or Plotly
- [ ] Try ensemble models (Random Forest, XGBoost) for comparison
- [ ] Deploy as a simple web app using Streamlit

---

## 👨‍💻 Author

**Arjuna Muhammad Arby Aljabar**

IBM Certified Data Analyst | Python for Data Science, AI & Development

[![Upwork](https://img.shields.io/badge/Upwork-Available-brightgreen?style=flat-square&logo=upwork)](https://www.upwork.com/freelancers/~019a7605087f863909)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*This project was completed as part of the IBM Data Analyst Professional Certificate on Coursera.*
