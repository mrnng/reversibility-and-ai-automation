import numpy as np
import pandas as pd
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

def perform_page_trend_test(dfs, value):
    # extract the domains
    df_domain_1, df_domain_2 = dfs

    # convert to wide dataframes with one row per participant, and keep only the states
    # and the reversibility rating
    wide_1 = df_domain_1.pivot(index="participant", columns="state", values=value)
    wide_2 = df_domain_2.pivot(index="participant", columns="state", values=value)

    # perform the Page test
    page_test_results = []

    for name, wide in [("file", wide_1), ("messaging", wide_2)]:
        L, z, p, R = page_trend_test(
            wide,
            ["S0", "S1", "S2", "S3"]
        )

        page_test_results.append({
            "domain": name,
            "S0_rank_sum": R[0],
            "S1_rank_sum": R[1],
            "S2_rank_sum": R[2],
            "S3_rank_sum": R[3],
            "Page_L": L,
            "z": z,
            "p": p
        })

    page_test_results = pd.DataFrame(page_test_results)
    return page_test_results
