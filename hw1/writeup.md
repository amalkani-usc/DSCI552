(d-i-A) Manhattan   k*=6  test_err=0.1100
  log10(p)=0.1  p=1.2589  test_err=0.0900
  log10(p)=0.2  p=1.5849  test_err=0.0900
  log10(p)=0.3  p=1.9953  test_err=0.0800
  log10(p)=0.4  p=2.5119  test_err=0.0800
  log10(p)=0.5  p=3.1623  test_err=0.0800
  log10(p)=0.6  p=3.9811  test_err=0.0600
  log10(p)=0.7  p=5.0119  test_err=0.0700
  log10(p)=0.8  p=6.3096  test_err=0.0800
  log10(p)=0.9  p=7.9433  test_err=0.0900
  log10(p)=1.0  p=10.0000  test_err=0.0900

  → Best log10(p) = 0.6  (p=3.9811)  test_err=0.0600  (k fixed at 6)
✓ Minkowski p plot saved

(d-i-C) Chebyshev   k*=16  test_err=0.0800

(d-ii)  Mahalanobis  k*=1  test_err=0.1700

── (d) Metric Comparison Table ──────────────────────────────
Metric                                      k*  Test Error
----------------------------------------------------------
Manhattan (p=1)                              6      0.1100
Minkowski best (log10p=0.6)                  6      0.0600
Chebyshev (p=inf)                           16      0.0800
Mahalanobis                                  1      0.1700

(e) Weighted euclidean     k*=6  test_err=0.1000

(e) Weighted manhattan     k*=26  test_err=0.1000

(e) Weighted chebyshev     k*=16  test_err=0.1100

── (e) Weighted Voting Summary ──────────────────────────────
Metric             k*  Best Test Error
--------------------------------------
euclidean           6           0.1000
manhattan          26           0.1000
chebyshev          16           0.1100

── (f) Lowest Training Error ────────────────────────────────
  k=1, Euclidean  →  train error = 0.0000
  (k=1 memorizes every training point → 0%)