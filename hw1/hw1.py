import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
import os

# ── Load data ──────────────────────────────────────────────────────────────
COLS = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle",
        "sacral_slope", "pelvic_radius", "degree_spondylolisthesis", "class"]

df = pd.read_csv("column_2C.dat", sep=r"\s+", header=None, names=COLS)
df["class_label"] = df["class"].map({"AB": 1, "NO": 0})
features = COLS[:-1]

palette   = {0: "#2196F3", 1: "#F44336"}
label_map = {0: "Class 0 – Normal", 1: "Class 1 – Abnormal"}
OUT = "."

# ─────────────────────────────────────────────────────────────────────────────
# 1i.  Pairwise Scatterplots
# ─────────────────────────────────────────────────────────────────────────────
pairs = list(combinations(features, 2))
ncols, nrows = 3, (len(pairs) + 2) // 3

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4.2))
fig.suptitle("Pairwise Scatterplots — Vertebral Column (2-Class)",
             fontsize=15, fontweight="bold", y=1.005)

for ax, (fx, fy) in zip(axes.flat, pairs):
    for cls, grp in df.groupby("class_label"):
        ax.scatter(grp[fx], grp[fy], c=palette[cls], label=label_map[cls],
                   alpha=0.55, s=20, edgecolors="none")
    ax.set_xlabel(fx.replace("_", " ").title(), fontsize=8)
    ax.set_ylabel(fy.replace("_", " ").title(), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.35)

for ax in axes.flat[len(pairs):]:
    ax.set_visible(False)

handles = [mpatches.Patch(color=palette[c], label=label_map[c]) for c in [0, 1]]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, -0.015), frameon=True)
plt.tight_layout()
plt.savefig(f"{OUT}/hw1_scatterplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ scatterplots saved")

# ─────────────────────────────────────────────────────────────────────────────
# 1ii.  Boxplots
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Boxplots by Class — Vertebral Column (2-Class)",
             fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flat, features):
    data_by_class = [df[df["class_label"] == c][feat].values for c in [0, 1]]
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, cls in zip(bp["boxes"], [0, 1]):
        patch.set_facecolor(palette[cls])
        patch.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Class 0\n(Normal)", "Class 1\n(Abnormal)"], fontsize=9)
    ax.set_title(feat.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.set_ylabel("Value", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/hw1_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ boxplots saved")

# ─────────────────────────────────────────────────────────────────────────────
# 1iii.  Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
class0 = df[df["class_label"] == 0].reset_index(drop=True)
class1 = df[df["class_label"] == 1].reset_index(drop=True)

train = pd.concat([class0.iloc[:70],  class1.iloc[:140]], ignore_index=True)
test  = pd.concat([class0.iloc[70:],  class1.iloc[140:]], ignore_index=True)

train.to_csv(f"{OUT}/hw1_train.csv", index=False)
test.to_csv(f"{OUT}/hw1_test.csv",  index=False)

print(f"\nDataset totals  →  Class 0: {len(class0)}  |  Class 1: {len(class1)}  |  Total: {len(df)}")
print(f"Train set       →  {len(train)} rows  (70 Class-0  +  140 Class-1)")
print(f"Test  set       →  {len(test)}  rows  ({len(class0)-70} Class-0  +  {len(class1)-140} Class-1)")

X_train, y_train = train[features].values, train["class_label"].values
X_test,  y_test  = test[features].values,  test["class_label"].values

# ─────────────────────────────────────────────────────────────────────────────
# 2ii.  KNN — train/test error vs k  (Euclidean)
# ─────────────────────────────────────────────────────────────────────────────
k_values = sorted(set(list(range(208, 3, -3)) + [1]), reverse=True)

train_errors, test_errors = [], []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    train_errors.append(1 - knn.score(X_train, y_train))
    test_errors.append(1 - knn.score(X_test,  y_test))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(k_values, train_errors, label="Train Error", color="#2196F3", linewidth=1.8)
ax.plot(k_values, test_errors,  label="Test Error",  color="#F44336", linewidth=1.8)
ax.set_xlabel("k", fontsize=12)
ax.set_ylabel("Error Rate", fontsize=12)
ax.set_title("KNN Train & Test Error vs k (Euclidean)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/hw1_knn_error_vs_k.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ error vs k plot saved")

best_idx = int(np.argmin(test_errors))
k_star   = k_values[best_idx]
print(f"\nk* = {k_star}  (test error = {test_errors[best_idx]:.4f})")

# ── Metrics at k* ──────────────────────────────────────────────────────────
knn_star = KNeighborsClassifier(n_neighbors=k_star, metric="euclidean")
knn_star.fit(X_train, y_train)
y_pred = knn_star.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

TPR  = TP / (TP + FN)
TNR  = TN / (TN + FP)
PREC = TP / (TP + FP)
F1   = 2 * PREC * TPR / (PREC + TPR)

print(f"\n── Metrics at k* = {k_star} ──────────────────")
print(f"Confusion Matrix:\n{cm}")
print(f"  TN={TN}  FP={FP}\n  FN={FN}  TP={TP}")
print(f"True Positive Rate (Recall) : {TPR:.4f}")
print(f"True Negative Rate (Spec.)  : {TNR:.4f}")
print(f"Precision                   : {PREC:.4f}")
print(f"F1-Score                    : {F1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2iii.  Learning Curve
# ─────────────────────────────────────────────────────────────────────────────
train_c0 = train[train["class_label"] == 0].reset_index(drop=True)
train_c1 = train[train["class_label"] == 1].reset_index(drop=True)

N_values    = list(range(10, 211, 10))
best_errors = []

for N in N_values:
    n0 = int(np.floor(N / 3))
    n1 = N - n0
    subset = pd.concat([train_c0.iloc[:n0], train_c1.iloc[:n1]], ignore_index=True)
    Xs, ys = subset[features].values, subset["class_label"].values
    k_candidates = list(range(1, N, 5)) or [1]

    best_err = 1.0
    for k in k_candidates:
        if k > len(Xs):
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(Xs, ys)
        err = 1 - knn.score(X_test, y_test)
        if err < best_err:
            best_err = err

    best_errors.append(best_err)
    print(f"  N={N:3d}  best test error = {best_err:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(N_values, best_errors, marker="o", markersize=4,
        color="#4CAF50", linewidth=1.8)
ax.set_xlabel("Training Set Size (N)", fontsize=12)
ax.set_ylabel("Best Test Error Rate", fontsize=12)
ax.set_title("Learning Curve — KNN (Euclidean)", fontsize=13, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT}/hw1_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ learning curve saved")

# ─────────────────────────────────────────────────────────────────────────────
# (d)  Alternative Metrics
# ─────────────────────────────────────────────────────────────────────────────
k_candidates = list(range(1, 197, 5))   # {1, 6, 11, ..., 196}

def best_k_and_error(metric, **kwargs):
    """Return (k*, best_test_error) over k_candidates for a given metric."""
    best_err, best_k = 1.0, 1
    for k in k_candidates:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric,
                                   metric_params=kwargs if kwargs else None)
        knn.fit(X_train, y_train)
        err = 1 - knn.score(X_test, y_test)
        if err < best_err:
            best_err, best_k = err, k
    return best_k, best_err

results = {}   # will hold all (k*, error) for summary table

# ── (d-i-A)  Manhattan  (Minkowski p=1) ────────────────────────────────────
k_manhattan, err_manhattan = best_k_and_error("manhattan")
results["Manhattan (p=1)"] = (k_manhattan, err_manhattan)
print(f"\n(d-i-A) Manhattan   k*={k_manhattan}  test_err={err_manhattan:.4f}")

# ── (d-i-B)  Minkowski with log10(p) in {0.1, 0.2, ..., 1.0} ──────────────
# Use k* found for Manhattan above
log10_p_values = [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)]
mink_errors = []

for lp in log10_p_values:
    p = 10 ** lp
    knn = KNeighborsClassifier(n_neighbors=k_manhattan, metric="minkowski",
                                p=p)
    knn.fit(X_train, y_train)
    err = 1 - knn.score(X_test, y_test)
    mink_errors.append(err)
    print(f"  log10(p)={lp:.1f}  p={p:.4f}  test_err={err:.4f}")

best_lp_idx  = int(np.argmin(mink_errors))
best_log10_p = log10_p_values[best_lp_idx]
best_mink_p  = 10 ** best_log10_p
results[f"Minkowski best (log10p={best_log10_p})"] = (k_manhattan, mink_errors[best_lp_idx])
print(f"\n  → Best log10(p) = {best_log10_p}  (p={best_mink_p:.4f})  "
      f"test_err={mink_errors[best_lp_idx]:.4f}  (k fixed at {k_manhattan})")

# Plot Minkowski error vs log10(p)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(log10_p_values, mink_errors, marker="o", color="#9C27B0", linewidth=1.8)
ax.axvline(best_log10_p, color="gray", linestyle="--", alpha=0.6,
           label=f"best log10(p)={best_log10_p}")
ax.set_xlabel("log10(p)", fontsize=12)
ax.set_ylabel("Test Error Rate", fontsize=12)
ax.set_title(f"Minkowski Error vs log10(p)  (k={k_manhattan})", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT}/hw1_minkowski_p.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Minkowski p plot saved")

# ── (d-i-C)  Chebyshev  (Minkowski p→∞) ───────────────────────────────────
k_cheby, err_cheby = best_k_and_error("chebyshev")
results["Chebyshev (p=inf)"] = (k_cheby, err_cheby)
print(f"\n(d-i-C) Chebyshev   k*={k_cheby}  test_err={err_cheby:.4f}")

# ── (d-ii)  Mahalanobis ────────────────────────────────────────────────────
# sklearn needs the inverse covariance matrix passed via metric_params
cov     = np.cov(X_train.T)
VI      = np.linalg.inv(cov)          # inverse covariance

best_err_maha, best_k_maha = 1.0, 1
for k in k_candidates:
    knn = KNeighborsClassifier(n_neighbors=k, metric="mahalanobis",
                                metric_params={"VI": VI})
    knn.fit(X_train, y_train)
    err = 1 - knn.score(X_test, y_test)
    if err < best_err_maha:
        best_err_maha, best_k_maha = err, k

results["Mahalanobis"] = (best_k_maha, best_err_maha)
print(f"\n(d-ii)  Mahalanobis  k*={best_k_maha}  test_err={best_err_maha:.4f}")

# ── (d) Summary Table ──────────────────────────────────────────────────────
print("\n── (d) Metric Comparison Table ──────────────────────────────")
print(f"{'Metric':<40} {'k*':>5}  {'Test Error':>10}")
print("-" * 58)
for name, (k, err) in results.items():
    print(f"{name:<40} {k:>5}  {err:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# (e)  Weighted Voting  (weight = 1/distance)
# ─────────────────────────────────────────────────────────────────────────────
k_weighted = list(range(1, 197, 5))   # {1, 6, 11, ..., 196}
weighted_results = {}

for metric in ["euclidean", "manhattan", "chebyshev"]:
    best_err, best_k = 1.0, 1
    for k in k_weighted:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric,
                                    weights="distance")
        knn.fit(X_train, y_train)
        err = 1 - knn.score(X_test, y_test)
        if err < best_err:
            best_err, best_k = err, k
    weighted_results[metric] = (best_k, best_err)
    print(f"\n(e) Weighted {metric:<12}  k*={best_k}  test_err={best_err:.4f}")

print("\n── (e) Weighted Voting Summary ──────────────────────────────")
print(f"{'Metric':<15} {'k*':>5}  {'Best Test Error':>15}")
print("-" * 38)
for metric, (k, err) in weighted_results.items():
    print(f"{metric:<15} {k:>5}  {err:>15.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# (f)  Lowest Training Error
# ─────────────────────────────────────────────────────────────────────────────
# k=1 on training set always gives 0 train error (each point is its own neighbor)
knn_k1 = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
knn_k1.fit(X_train, y_train)
lowest_train_err = 1 - knn_k1.score(X_train, y_train)

print(f"\n── (f) Lowest Training Error ────────────────────────────────")
print(f"  k=1, Euclidean  →  train error = {lowest_train_err:.4f}")
print("  (k=1 memorizes every training point → 0% train error by definition)")