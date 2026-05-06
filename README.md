# 📊 High-Dimensional Feature Selection and Regularization Analysis

> **Empirical study of L1 (Lasso) and L2 (Ridge) regularization in high-dimensional settings** — evaluating sparsity, coefficient stability, and predictive performance across structured and unstructured data.

---

## 🧠 What This Project Studies

In machine learning, when the number of features (p) approaches or exceeds the number of samples (n), standard regression breaks down. **Regularization** adds a penalty to prevent overfitting. This project empirically answers:

| Question | Method Used |
|---|---|
| Which method produces sparser models? | Sparsity vs. alpha analysis |
| Which method is more stable? | Bootstrap resampling (50 iterations) |
| Which generalizes better? | 5-fold cross-validation |
| How does correlation between features affect each? | Structured block correlation experiment |

---

## 📐 Mathematical Background

**Lasso (L1):**
$$\hat{\beta}^{Lasso} = \arg\min_\beta \|y - X\beta\|_2^2 + \alpha \|\beta\|_1$$

Promotes **exact zeros** — performs automatic feature selection.

**Ridge (L2):**
$$\hat{\beta}^{Ridge} = \arg\min_\beta \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2$$

Shrinks all coefficients — **never** drops features entirely.

---

## 🔬 Experiments

### 1. Sparsity Analysis
- 200 features, only 20 truly informative
- Lasso drives irrelevant coefficients to **exactly 0** at moderate α
- Ridge retains all features regardless of α

### 2. Coefficient Stability (Bootstrap)
- 50 bootstrap resamples
- Lasso coefficients vary more (high variance), especially near threshold
- Ridge coefficients are **smoother and more stable**

### 3. Predictive Performance
- 5-fold cross-validation over 60 values of α (log-spaced)
- Both methods compared on MSE and R²
- Optimal α identified for each

### 4. Structured Dependence (Correlated Features)
- 5 groups × 10 correlated features (ρ = 0.85)
- **Lasso arbitrarily picks one feature per group** (sparse but unstable)
- **Ridge spreads weight across correlated features** (interpretable group effect)

---

## 📈 Results Summary

| Metric | Lasso (L1) | Ridge (L2) |
|---|---|---|
| Sparsity | ✅ Exact zeros | ❌ Dense |
| Stability | ⚠️ Moderate | ✅ High |
| Correlated features | ⚠️ Picks arbitrarily | ✅ Distributes weight |
| Interpretability | ✅ Fewer features | ⚠️ All features retained |
| Best for | n << p, sparse truth | Multicollinear settings |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/anas-git007/regularization_analysis
cd regularization_analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python regularization_analysis.py
```

Output: `regularization_analysis.png` — a 7-panel figure covering all experiments.

---


## 📦 Dependencies

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
```

---

## 💡 Key Takeaways

- **Lasso = feature selector**: In truly sparse settings, Lasso recovers the correct support
- **Ridge = stabilizer**: When features are correlated, Ridge distributes weight more fairly
- The **robustness–interpretability trade-off** is not just theoretical — it is clearly visible in bootstrap variance and group-correlation experiments
- Optimal α is data-dependent and **must be tuned via cross-validation**

---

## 🔗 Related Work

- Tibshirani (1996) — *Regression Shrinkage and Selection via the Lasso*
- Hoerl & Kennard (1970) — *Ridge Regression: Biased Estimation for Nonorthogonal Problems*
- Hastie, Tibshirani & Friedman — *Elements of Statistical Learning*, Chapter 3
