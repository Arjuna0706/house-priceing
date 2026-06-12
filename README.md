# 🏠 House Sales Price Prediction — King County, USA

> Predicting residential home prices in King County (Seattle metro area) using regression models, built as the capstone project for **IBM's *Data Analysis with Python*** course on Coursera.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Overview

This project develops a machine learning model to estimate the **market price of houses** in King County, Washington — the county that contains Seattle. Using a dataset of **21,613 home sales** recorded between May 2014 and May 2015, the analysis walks through the complete data-science workflow: from cleaning raw data, to exploratory analysis, to fitting and tuning regression models.

The goal is to answer a practical question: **which features of a home most strongly drive its sale price, and how accurately can we predict that price from them?**

---

## 🎯 Objectives

- Clean and prepare a real-world housing dataset (handle missing values, drop noise columns).
- Explore relationships between home attributes and price through statistics and visualization.
- Build regression models of increasing sophistication and compare their performance.
- Apply regularization and pipelines to combat overfitting and improve generalization.

---

## 📊 Dataset

The dataset (`kc_house_data_NaN.csv`) contains **21,613 records** and **22 columns**. Each row is a single home sale. Key features include:

| Feature | Description |
|---|---|
| `price` | **Target variable** — sale price of the home (USD) |
| `sqft_living` | Interior living space (square feet) |
| `grade` | Overall construction & design grade (1–13) |
| `bedrooms` / `bathrooms` | Number of bedrooms / bathrooms |
| `sqft_above` | Square footage above ground level |
| `sqft_living15` | Living space of the 15 nearest neighbors |
| `waterfront` | Whether the home overlooks the waterfront (0/1) |
| `view` | Quality of the view (0–4) |
| `floors`, `condition`, `yr_built`, `zipcode`, `lat`, `long` | Additional structural & locational attributes |

**Data quality note:** The raw file contained missing values in `bedrooms` (13) and `bathrooms` (10), plus two non-informative index columns (`Unnamed: 0`, `id`) that were removed during cleaning.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data handling** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Modeling** | scikit-learn (`LinearRegression`, `Ridge`, `PolynomialFeatures`, `Pipeline`, `StandardScaler`) |
| **Environment** | Jupyter Notebook |

---

## 🔍 Workflow & Methodology

### 1. Data Cleaning
- Dropped the redundant `Unnamed: 0` and `id` columns.
- Imputed missing `bedrooms` and `bathrooms` values with their respective column means, eliminating all NaNs.

### 2. Exploratory Data Analysis (EDA)
- **Boxplot** of `price` vs `waterfront` — waterfront homes command a clearly higher and wider price range.
- **Regression plot** of `price` vs `sqft_above` — confirms a strong positive linear trend.
- **Correlation analysis** against `price` to rank the most predictive features.

### 3. Modeling
Models were built progressively to demonstrate the impact of feature richness and regularization:

| Step | Model | Features | R² Score |
|---|---|---|:---:|
| 1 | Simple Linear Regression | `long` only | 0.0005 |
| 2 | Simple Linear Regression | `sqft_living` only | 0.493 |
| 3 | Multiple Linear Regression | All features | 0.700 |
| 4 | Polynomial Pipeline (scale → poly → linear) | All features | **0.830** |
| 5 | Ridge Regression (α=0.1) | All features (test set) | 0.690 |
| 6 | Ridge + 2nd-degree Polynomial (test set) | All features | **0.795** |

> Models 1–4 are evaluated in-sample (full data); models 5–6 use a proper **85/15 train-test split** (18,371 train / 3,242 test) for honest generalization estimates.

---

## 💡 Key Insights

- **Living space dominates.** `sqft_living` (corr ≈ **0.70**) and `grade` (corr ≈ **0.67**) are the single strongest predictors of price — buyers pay for space and build quality above all else.
- **Location alone is weak.** Longitude alone explains essentially **none** of the variance (R² ≈ 0.0005). Geography matters, but only in combination with structural features — raw coordinates are poor standalone predictors.
- **More features ≫ one feature.** Moving from a single-variable model to a full multiple-regression model lifted R² from **0.49 → 0.70**, a roughly 40% improvement in explanatory power.
- **Non-linearity helps.** A polynomial pipeline pushed in-sample R² to **0.83**, capturing curved relationships that a plain linear model misses.
- **Regularization keeps it honest.** On unseen test data, the Ridge + polynomial model achieves **R² ≈ 0.795** — strong out-of-sample performance that demonstrates the model generalizes rather than memorizes.

---

## 📈 Results Summary

The final tuned model (**Ridge Regression with 2nd-degree polynomial features and standardized inputs**) explains approximately **80% of the variance in house prices on unseen data**, making it a solid baseline for price estimation in this market.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the notebook
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook House_Priceing.ipynb
```

> **Note:** The notebook reads `kc_house_data_NaN.csv`. Make sure the data file sits in the same directory as the notebook.

---

## 📁 Repository Structure

```
.
├── House_Priceing.ipynb      # Main analysis notebook
├── kc_house_data.csv      # King County housing dataset
└── README.md                  # Project documentation
```

---

## 🔮 Future Improvements

- Engineer location-aware features (e.g., price-per-zipcode, distance to city center).
- Test ensemble methods (Random Forest, Gradient Boosting / XGBoost) for likely accuracy gains.
- Apply log-transformation to `price` to address right-skew and stabilize residuals.
- Add cross-validated hyperparameter tuning (`GridSearchCV`) for the Ridge α and polynomial degree.

---

## 🙏 Acknowledgements

- **Dataset:** King County House Sales, originally published on Kaggle.
- **Course:** *Data Analysis with Python* — IBM, via Coursera.

---

## 📄 License

This project is released under the **MIT License** — feel free to use, modify, and learn from it.
