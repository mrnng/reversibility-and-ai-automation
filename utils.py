import numpy as np
import pandas as pd
from scipy import stats

def page_trend_test(wide, states, direction="increasing"):
    data = wide[states].dropna().to_numpy()
    n, k = data.shape

    ranks = np.array([ stats.rankdata(row, method="average") for row in data ])

    R = ranks.sum(axis=0)
    weights = np.arange(1, k + 1)
    L = np.sum(weights * R)
    expected_L = n * k * (k + 1)**2 / 4
    variance_L = n * k**2 * (k + 1) * (k**2 - 1) / 144
    z = (L - expected_L) / np.sqrt(variance_L)

    if direction == "increasing":
        p = stats.norm.sf(z)
    elif direction == "decreasing":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("Direction must be 'increasing' or 'decreasing'.")

    return L, z, p, R

def perform_page_trend_test(dfs, value, direction):
    # extract the domains
    df_domain_1, df_domain_2 = dfs

    # convert to wide dataframes with one row per participant, and keep only the states
    # and the chosen value
    wide_1 = df_domain_1.pivot(index="participant", columns="state", values=value)
    wide_2 = df_domain_2.pivot(index="participant", columns="state", values=value)

    # perform the Page test
    page_test_results = []

    for name, wide in [("file", wide_1), ("messaging", wide_2)]:
        L, z, p, R = page_trend_test(wide, ["S0", "S1", "S2-H", "S2-A"], direction=direction)

        page_test_results.append({
            "domain": name,
            "S0_rank_sum": R[0],
            "S1_rank_sum": R[1],
            "S2-H_rank_sum": R[2],
            "S2-A_rank_sum": R[3],
            "Page_L": L,
            "z": z,
            "p": p
        })

    page_test_results = pd.DataFrame(page_test_results)
    return page_test_results

def friedman_test(wide, states):
    data = wide[states].dropna().to_numpy()
    n, k = data.shape
 
    chi2, p = stats.friedmanchisquare(*[data[:, j] for j in range(k)]) 
    W = chi2 / (n * (k - 1))
 
    return chi2, p, W, n

def perform_friedman_test(dfs, value):
    # extract the domains
    df_domain_1, df_domain_2 = dfs

    # convert to wide dataframes with one row per participant, and keep only the states
    # and the chosen value
    wide_1 = df_domain_1.pivot(index="participant", columns="state", values=value)
    wide_2 = df_domain_2.pivot(index="participant", columns="state", values=value)
 
    # perform the Friedman test
    friedman_test_results = []
    for name, wide in [("file", wide_1), ("messaging", wide_2)]:
        chi2, p, W, n = friedman_test(wide, ["S0", "S1", "S2-H", "S2-A"])

        friedman_test_results.append({
            "domain": name,
            "n": n,
            "chi2": chi2,
            "p": p,
            "Kendall_W": W
        })
 
    friedman_test_results = pd.DataFrame(friedman_test_results)
    return friedman_test_results

def holm_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted

def rank_biserial(diff):
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff), method="average")
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    return (w_pos - w_neg) / (w_pos + w_neg)

def bootstrap_rank_biserial(diff, n_boot=5000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([rank_biserial(diff[i]) for i in idx])
    return np.quantile(boots, [alpha / 2, 1 - alpha / 2])

def pairwise_wilcoxon(wide, states, n_boot=5000, seed=0):
    data = wide[states].dropna()
    rows = []
 
    for a, b in zip(states[:-1], states[1:]):
        x = data[a].to_numpy()
        y = data[b].to_numpy()
        diff = y - x
 
        if np.all(diff == 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
 
        r = rank_biserial(diff)
        ci_low, ci_high = bootstrap_rank_biserial(diff, n_boot=n_boot, seed=seed)
 
        rows.append({
            "comparison": f"{a}->{b}",
            "n": len(diff),
            "median_diff": np.median(diff),
            "W": stat,
            "p": p,
            "rank_biserial": r,
            "CI_low": ci_low,
            "CI_high": ci_high
        })
 
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p"].to_numpy())
    return out

def perform_pairwise_tests(dfs, value, n_boot=5000):
    # extract the domains
    df_domain_1, df_domain_2 = dfs

    # convert to wide dataframes with one row per participant, and keep only the states
    # and the chosen value
    wide_1 = df_domain_1.pivot(index="participant", columns="state", values=value)
    wide_2 = df_domain_2.pivot(index="participant", columns="state", values=value)
 
    # perform the pairwise Wilcoxon tests
    pairwise_test_results = []

    for name, wide in [("file", wide_1), ("messaging", wide_2)]:
        result = pairwise_wilcoxon(wide, ["S0", "S1", "S2-H", "S2-A"], n_boot=n_boot)
        result.insert(0, "domain", name)
        pairwise_test_results.append(result)
 
    pairwise_test_results = pd.concat(pairwise_test_results, ignore_index=True)
    return pairwise_test_results
