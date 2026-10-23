import numpy as np
from scipy import stats

def page_trend_test(wide, states):
    data = wide[states].dropna().to_numpy()
    n, k = data.shape

    ranks = np.array([
        stats.rankdata(row, method="average")
        for row in data
    ])

    R = ranks.sum(axis=0)
    weights = np.arange(1, k + 1)
    L = np.sum(weights * R)
    expected_L = n * k * (k + 1)**2 / 4
    variance_L = (
        n * k**2 * (k + 1) * (k**2 - 1) / 144
    )
    z = (L - expected_L) / np.sqrt(variance_L)
    p = stats.norm.sf(z)

    return L, z, p, R
