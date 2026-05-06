"""
High-Dimensional Feature Selection and Regularization Analysis
==============================================================
Empirical comparison of L1 (Lasso) and L2 (Ridge) regularization in high-dimensional settings.
Evaluates sparsity, coefficient stability, and predictive performance.

Author: [Muhammad Anas]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
import warnings
import os
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#e6edf3',
    'grid.color': '#21262d',
    'axes.edgecolor': '#30363d',
})

LASSO_COLOR = '#58a6ff'
RIDGE_COLOR = '#f78166'
ACCENT_GREEN = '#3fb950'
ACCENT_YELLOW = '#d29922'


# ── Data Generation ───────────────────────────────────────────────────────────
def generate_high_dimensional_data(n_samples=300, n_features=200, n_informative=20,
                                    noise=0.5, random_state=42):
    """
    Generate synthetic high-dimensional regression data.
    Only n_informative features truly affect the target — the rest are noise.
    This mimics real-world genomics, finance, or text data.
    """
    X, y, true_coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, true_coef


# ── Sparsity Analysis ────────────────────────────────────────────────────────
def analyze_sparsity(X_train, y_train, alphas):
    """
    Track how many coefficients are exactly zero (Lasso) vs near-zero (Ridge)
    across a range of regularization strengths (alpha values).
    """
    lasso_nonzero = []
    ridge_nonzero = []
    lasso_coefs = []
    ridge_coefs = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train)
        ridge = Ridge(alpha=alpha).fit(X_train, y_train)

        lasso_nonzero.append(np.sum(lasso.coef_ != 0))
        ridge_nonzero.append(np.sum(np.abs(ridge.coef_) > 1e-4))
        lasso_coefs.append(lasso.coef_.copy())
        ridge_coefs.append(ridge.coef_.copy())

    return (np.array(lasso_nonzero), np.array(ridge_nonzero),
            np.array(lasso_coefs), np.array(ridge_coefs))


# ── Coefficient Stability ────────────────────────────────────────────────────
def analyze_stability(X, y, n_bootstraps=50, alpha_lasso=0.01, alpha_ridge=1.0,
                       random_state=42):
    """
    Bootstrap resampling: fit Lasso and Ridge on 50 different random subsets.
    Measure how much coefficients vary → lower std = more stable.
    """
    rng = np.random.RandomState(random_state)
    n_features = X.shape[1]
    lasso_coefs = np.zeros((n_bootstraps, n_features))
    ridge_coefs = np.zeros((n_bootstraps, n_features))

    for i in range(n_bootstraps):
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_b, y_b = X[idx], y[idx]
        lasso_coefs[i] = Lasso(alpha=alpha_lasso, max_iter=10000).fit(X_b, y_b).coef_
        ridge_coefs[i] = Ridge(alpha=alpha_ridge).fit(X_b, y_b).coef_

    return lasso_coefs, ridge_coefs


# ── Predictive Performance ───────────────────────────────────────────────────
def evaluate_performance(X, y, alphas, cv=5):
    """
    K-fold cross-validation: compare test MSE and R² for both methods
    across a range of alphas to find the optimal regularization strength.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    lasso_mse, ridge_mse = [], []
    lasso_r2, ridge_r2 = [], []

    for alpha in alphas:
        lm, rm, lr, rr = [], [], [], []
        for train_idx, test_idx in kf.split(X):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            lasso = Lasso(alpha=alpha, max_iter=10000).fit(Xtr, ytr)
            ridge = Ridge(alpha=alpha).fit(Xtr, ytr)

            lm.append(mean_squared_error(yte, lasso.predict(Xte)))
            rm.append(mean_squared_error(yte, ridge.predict(Xte)))
            lr.append(r2_score(yte, lasso.predict(Xte)))
            rr.append(r2_score(yte, ridge.predict(Xte)))

        lasso_mse.append(np.mean(lm))
        ridge_mse.append(np.mean(rm))
        lasso_r2.append(np.mean(lr))
        ridge_r2.append(np.mean(rr))

    return (np.array(lasso_mse), np.array(ridge_mse),
            np.array(lasso_r2), np.array(ridge_r2))


# ── Structured Dependence ────────────────────────────────────────────────────
def analyze_correlated_features(n_samples=300, n_groups=5, group_size=10,
                                  correlation=0.85, random_state=42):
    """
    When features are correlated (common in real data), Lasso arbitrarily picks
    one from each group; Ridge distributes weight across all correlated features.
    This highlights the robustness–interpretability trade-off.
    """
    rng = np.random.RandomState(random_state)
    n_features = n_groups * group_size
    X = np.zeros((n_samples, n_features))

    for g in range(n_groups):
        base = rng.randn(n_samples)
        for f in range(group_size):
            noise = rng.randn(n_samples)
            X[:, g * group_size + f] = (np.sqrt(correlation) * base +
                                         np.sqrt(1 - correlation) * noise)

    true_coef = np.zeros(n_features)
    for g in range(n_groups):
        true_coef[g * group_size] = rng.choice([-2, 2])

    y = X @ true_coef + 0.3 * rng.randn(n_samples)
    X = StandardScaler().fit_transform(X)

    lasso = Lasso(alpha=0.05, max_iter=10000).fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)

    return true_coef, lasso.coef_, ridge.coef_, n_groups, group_size


# ── Visualization ────────────────────────────────────────────────────────────
def plot_all_results(X, y, true_coef, alphas, results):
    (lasso_nonzero, ridge_nonzero, lasso_coef_path, ridge_coef_path,
     lasso_boot, ridge_boot, lasso_mse, ridge_mse, lasso_r2, ridge_r2,
     true_corr, lasso_corr, ridge_corr, n_groups, group_size) = results

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('High-Dimensional Regularization Analysis:\nL1 (Lasso) vs L2 (Ridge)',
                 fontsize=18, fontweight='bold', color='#e6edf3', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Sparsity vs Alpha ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(alphas, lasso_nonzero, color=LASSO_COLOR, lw=2.5, label='Lasso (L1)', marker='o', ms=3)
    ax1.semilogx(alphas, ridge_nonzero, color=RIDGE_COLOR, lw=2.5, label='Ridge (L2)', marker='s', ms=3)
    ax1.axhline(y=20, color=ACCENT_GREEN, ls='--', lw=1.5, label='True informative (20)', alpha=0.8)
    ax1.set_xlabel('Regularization strength α')
    ax1.set_ylabel('Non-zero coefficients')
    ax1.set_title('① Sparsity vs Regularization')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Coefficient Path (Lasso) ──
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(lasso_coef_path.shape[1]):
        color = ACCENT_GREEN if true_coef[i] != 0 else '#30363d'
        alpha_val = 0.8 if true_coef[i] != 0 else 0.2
        ax2.semilogx(alphas, lasso_coef_path[:, i], color=color, alpha=alpha_val, lw=1)
    ax2.set_xlabel('Regularization strength α')
    ax2.set_ylabel('Coefficient value')
    ax2.set_title('② Lasso Coefficient Paths\n(green = truly informative)')
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Coefficient Path (Ridge) ──
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(ridge_coef_path.shape[1]):
        color = ACCENT_GREEN if true_coef[i] != 0 else '#30363d'
        alpha_val = 0.8 if true_coef[i] != 0 else 0.2
        ax3.semilogx(alphas, ridge_coef_path[:, i], color=color, alpha=alpha_val, lw=1)
    ax3.set_xlabel('Regularization strength α')
    ax3.set_ylabel('Coefficient value')
    ax3.set_title('③ Ridge Coefficient Paths\n(green = truly informative)')
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Bootstrap Stability ──
    ax4 = fig.add_subplot(gs[1, 0])
    lasso_std = np.std(lasso_boot, axis=0)
    ridge_std = np.std(ridge_boot, axis=0)
    x_pos = np.arange(len(lasso_std))
    ax4.fill_between(x_pos, 0, lasso_std, alpha=0.6, color=LASSO_COLOR, label='Lasso std')
    ax4.fill_between(x_pos, 0, ridge_std, alpha=0.6, color=RIDGE_COLOR, label='Ridge std')
    ax4.set_xlabel('Feature index')
    ax4.set_ylabel('Coefficient std (50 bootstraps)')
    ax4.set_title('④ Coefficient Stability')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: MSE vs Alpha ──
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogx(alphas, lasso_mse, color=LASSO_COLOR, lw=2.5, label='Lasso')
    ax5.semilogx(alphas, ridge_mse, color=RIDGE_COLOR, lw=2.5, label='Ridge')
    ax5.set_xlabel('Regularization strength α')
    ax5.set_ylabel('Cross-validated MSE')
    ax5.set_title('⑤ Predictive Performance (MSE)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: R² vs Alpha ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogx(alphas, lasso_r2, color=LASSO_COLOR, lw=2.5, label='Lasso')
    ax6.semilogx(alphas, ridge_r2, color=RIDGE_COLOR, lw=2.5, label='Ridge')
    ax6.set_xlabel('Regularization strength α')
    ax6.set_ylabel('Cross-validated R²')
    ax6.set_title('⑥ Predictive Performance (R²)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ── Panel 7: Correlated Features ──
    ax7 = fig.add_subplot(gs[2, :])
    n_features_corr = n_groups * group_size
    x = np.arange(n_features_corr)
    width = 0.3
    bars_true = ax7.bar(x - width, true_corr, width, color=ACCENT_GREEN, alpha=0.7, label='True coefficients')
    bars_lasso = ax7.bar(x, lasso_corr, width, color=LASSO_COLOR, alpha=0.7, label='Lasso estimates')
    bars_ridge = ax7.bar(x + width, ridge_corr, width, color=RIDGE_COLOR, alpha=0.7, label='Ridge estimates')
    for g in range(n_groups - 1):
        ax7.axvline(x=(g + 1) * group_size - 0.5, color='#8b949e', ls=':', lw=1, alpha=0.6)
    ax7.set_xlabel('Feature index (grouped by correlation block)')
    ax7.set_ylabel('Coefficient value')
    ax7.set_title('⑦ Structured Dependence: Robustness–Interpretability Trade-off\n'
                  '(Within each block, features are highly correlated — Lasso picks one, Ridge spreads weight)')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')

    plt.savefig('regularization_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("✓ Saved: regularization_analysis.png")
    plt.close()


# ── Summary Table ─────────────────────────────────────────────────────────────
def print_summary(lasso_mse, ridge_mse, lasso_r2, ridge_r2,
                  lasso_nonzero, ridge_nonzero, alphas):
    best_l = np.argmin(lasso_mse)
    best_r = np.argmin(ridge_mse)
    print("\n" + "="*60)
    print("  REGULARIZATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Metric':<30} {'Lasso (L1)':>12} {'Ridge (L2)':>12}")
    print("-"*60)
    print(f"{'Best CV MSE':<30} {lasso_mse[best_l]:>12.3f} {ridge_mse[best_r]:>12.3f}")
    print(f"{'Best CV R²':<30} {lasso_r2[best_l]:>12.3f} {ridge_r2[best_r]:>12.3f}")
    print(f"{'Best alpha':<30} {alphas[best_l]:>12.4f} {alphas[best_r]:>12.4f}")
    print(f"{'Non-zero coefs @ best α':<30} {lasso_nonzero[best_l]:>12} {ridge_nonzero[best_r]:>12}")
    print("="*60)
    print("\nKey Insights:")
    print("  • Lasso achieves sparsity: drives irrelevant features to exactly 0")
    print("  • Ridge shrinks all coefficients but retains all features")
    print("  • In high-dimensional settings, Lasso is preferred for interpretability")
    print("  • Ridge is more stable under correlated/multicollinear features")
    print("  • Trade-off: sparsity (Lasso) vs stability (Ridge)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating high-dimensional data (300 samples, 200 features, 20 informative)...")
    X, y, true_coef = generate_high_dimensional_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alphas = np.logspace(-3, 2, 60)

    print("Analyzing sparsity across alpha range...")
    lasso_nonzero, ridge_nonzero, lasso_coef_path, ridge_coef_path = analyze_sparsity(X_train, y_train, alphas)

    print("Running bootstrap stability analysis (50 resamples)...")
    lasso_boot, ridge_boot = analyze_stability(X, y)

    print("Evaluating cross-validated predictive performance...")
    lasso_mse, ridge_mse, lasso_r2, ridge_r2 = evaluate_performance(X, y, alphas)

    print("Analyzing structured dependence / correlated features...")
    true_corr, lasso_corr, ridge_corr, n_groups, group_size = analyze_correlated_features()

    print("Generating visualization...")
    results = (lasso_nonzero, ridge_nonzero, lasso_coef_path, ridge_coef_path,
               lasso_boot, ridge_boot, lasso_mse, ridge_mse, lasso_r2, ridge_r2,
               true_corr, lasso_corr, ridge_corr, n_groups, group_size)
    plot_all_results(X, y, true_coef, alphas, results)

    print_summary(lasso_mse, ridge_mse, lasso_r2, ridge_r2, lasso_nonzero, ridge_nonzero, alphas)
    print("\nDone! Check regularization_analysis.png for the full report.")
    print("PNG saved to:", os.path.abspath("regularization_analysis.png"))

if __name__ == "__main__":
    main()
