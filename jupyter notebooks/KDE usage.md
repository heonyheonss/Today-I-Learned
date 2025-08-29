# Kernel Density Estimation, KDE
## what i understood
replace a datapoint to a gaussian distribution,
and then summing pointwise all of the distributions

## (1) sklearn : from sklearn.neighbors import KernelDensity
    bandwidth = 0.35  # bandwidth could be controlled by distribution
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(y)
    
    y_grid = np.linspace(y.min() - 1, y.max() + 1, 500).reshape(-1, 1)
    log_density = kde.score_samples(y_grid)
    density = np.exp(log_density)  # log density → density
