import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.formula.api as smf
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

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


def prepare_repeated_data(df):
    # check that each participant has all eight tasks, with no duplicate rows
    df = df.copy()
    outcomes = ["reversibility", "willingness", "risk", "bother"]
    required = ["participant", "task"] + outcomes

    if df.empty or df[required].isna().any().any():
        raise ValueError("The ratings must not be empty or contain missing values.")
    if df.duplicated(["participant", "task"]).any():
        raise ValueError("Duplicate participant and task combinations were found.")
    for participant, data in df.groupby("participant"):
        if set(data["task"]) != set(range(1, 9)):
            raise ValueError(f"Participant {participant} does not have all eight tasks.")
    for value in outcomes:
        if not df[value].between(1, 7).all() or not (df[value] % 1 == 0).all():
            raise ValueError(f"{value} must contain integer ratings from 1 to 7.")

    # map each task to its state and domain, with explicit reference categories
    states = ["S0", "S1", "S2-H", "S2-A"]
    state_map = dict(zip(range(1, 9), states * 2))
    df["state"] = pd.Categorical(df["task"].map(state_map), categories=states)
    df["domain"] = pd.Categorical(
        np.where(df["task"] <= 4, "file", "messaging"),
        categories=["file", "messaging"]
    )
    return df

def mixed_effects_model(df, value):
    # fit a separate Gaussian model for each outcome, treating state as categorical
    # the participant random intercept accounts for repeated ratings
    # this approximates the 1-7 ordinal scale as an interval scale
    model = smf.mixedlm(
        f"{value} ~ state * domain", df,
        groups=df["participant"], re_formula="1"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(reml=True, method="powell", maxiter=2000)

    if not result.converged:
        raise RuntimeError(f"The mixed-effects model did not converge for {value}.")

    model_warnings = [str(warning.message) for warning in caught]
    return result, model_warnings

def perform_mixed_effects_model(df, value):
    # fit the model and collect its fixed effects and variance components
    model, model_warnings = mixed_effects_model(df, value)
    ci = model.conf_int().loc[model.fe_params.index]
    fixed_effects = []

    for term in model.fe_params.index:
        fixed_effects.append({
            "term": term,
            "estimate": model.fe_params[term],
            "se": model.bse_fe[term],
            "CI_low": ci.loc[term, 0],
            "CI_high": ci.loc[term, 1],
            "p_wald": model.pvalues[term]
        })

    model_results = pd.DataFrame([{
        "outcome": value,
        "n": df["participant"].nunique(),
        "observations": len(df),
        "converged": model.converged,
        "participant_variance": model.cov_re.iloc[0, 0],
        "residual_variance": model.scale,
        "warnings": "; ".join(model_warnings)
    }])
    fixed_effects = pd.DataFrame(fixed_effects)
    return model, fixed_effects, model_results

def mixed_effects_contrasts(df, model, value, n_boot=20000, seed=20261104):
    # predict each state in each domain using the fitted model
    states = ["S0", "S1", "S2-H", "S2-A"]
    cells = pd.DataFrame([
        {"state": state, "domain": domain}
        for state in states for domain in ["file", "messaging"]
    ])
    predictions = np.asarray(model.predict(cells)).reshape(4, 2)
    predicted_means = predictions.mean(axis=1)

    # average the two domains within each participant
    wide = df.pivot(index="participant", columns=["domain", "state"], values=value)
    pooled = np.column_stack([
        (wide[("file", state)] + wide[("messaging", state)]) / 2
        for state in states
    ])
    n = len(wide)

    # resample whole participants, preserving their eight related ratings
    # these pointwise intervals and participant t tests allow different state
    # responses across participants; they are separate from model Wald tests
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    differences = np.diff(pooled, axis=1)
    estimates = np.diff(predicted_means)
    comparisons = [f"{a}->{b}" for a, b in zip(states[:-1], states[1:])]

    # compare the middle step with each neighboring step as exploratory checks
    differences = np.column_stack([
        differences,
        differences[:, 1] - differences[:, 0],
        differences[:, 1] - differences[:, 2]
    ])
    estimates = np.append(estimates, [estimates[1] - estimates[0], estimates[1] - estimates[2]])
    comparisons += ["middle_minus_first", "middle_minus_last"]
    rows = []

    for j, comparison in enumerate(comparisons):
        diff = differences[:, j]

        # equality holds because all participants have every state in both domains
        if not np.isclose(estimates[j], diff.mean(), atol=1e-8):
            raise ValueError("The model and participant contrast estimates do not agree.")

        ci_low, ci_high = np.quantile(diff[idx].mean(axis=1), [0.025, 0.975])
        test = stats.ttest_1samp(diff, 0)
        rows.append({
            "outcome": value,
            "comparison": comparison,
            "n": n,
            "estimate": estimates[j],
            "se_cluster": diff.std(ddof=1) / np.sqrt(n),
            "CI_low": ci_low,
            "CI_high": ci_high,
            "t": test.statistic,
            "df": n - 1,
            "p": test.pvalue
        })

    return pd.DataFrame(rows)

def correct_mixed_effects_contrasts(results):
    # adjust the three adjacent willingness tests as the primary family
    # adjust the other nine adjacent tests as one secondary family
    # adjust the eight exploratory middle-step comparisons as another family
    results = results.copy()
    adjacent = results["comparison"].str.contains("->")
    primary = adjacent & (results["outcome"] == "willingness")
    secondary = adjacent & (results["outcome"] != "willingness")

    for mask in [primary, secondary, ~adjacent]:
        results.loc[mask, "p_holm"] = holm_correction(results.loc[mask, "p"].to_numpy())

    return results

def domain_state_interaction(df, value):
    # test the three state-by-domain contrasts at the participant level
    # Hotelling's test allows arbitrary covariance among these contrasts
    # its F reference is approximate for these bounded ordinal ratings
    states = ["S0", "S1", "S2-H", "S2-A"]
    wide = df.pivot(index="participant", columns=["domain", "state"], values=value)
    domain_diff = np.column_stack([
        wide[("messaging", state)] - wide[("file", state)] for state in states
    ])
    diff = domain_diff[:, 1:] - domain_diff[:, [0]]
    n, k = diff.shape
    mean = diff.mean(axis=0)
    cov = np.cov(diff, rowvar=False)

    if np.linalg.matrix_rank(cov) != k:
        raise ValueError("The interaction contrast covariance is singular.")

    T2 = n * mean @ np.linalg.solve(cov, mean)
    F = (n - k) / (k * (n - 1)) * T2
    p = stats.f.sf(F, k, n - k)
    return {"outcome": value, "n": n, "F": F, "df1": k, "df2": n - k, "p": p}

def rating_quality_flags(df):
    # flag invariant profiles and later-numbered copies of identical profiles
    # these are sensitivity checks, not proof that a response is invalid
    outcomes = ["willingness", "risk", "bother", "reversibility"]
    wide = df.pivot(index="participant", columns="task", values=outcomes).sort_index()
    invariant = []

    for participant, data in df.groupby("participant"):
        if all(data[value].nunique() == 1 for value in outcomes):
            invariant.append(participant)

    duplicate_later = wide.index[wide.duplicated(keep="first")].tolist()
    excluded = sorted(set(invariant + duplicate_later))
    return invariant, duplicate_later, excluded

def participant_summary(df):
    # count people once rather than counting their eight vignette rows
    people = df.drop_duplicates("participant")
    columns = ["age", "gender", "education", "technical_background", "ai_usage_frequency", "ai_agent_use"]
    rows = []

    for column in columns:
        counts = people[column].value_counts(dropna=False)
        for category, count in counts.items():
            rows.append({
                "measure": column,
                "category": category,
                "n": count,
                "percent": 100 * count / len(people)
            })

    return pd.DataFrame(rows)

def perform_prediction(df, value):
    # use the same responses and participant folds for every feature set
    data = df.dropna(subset=["participant", "reversibility", value])
    groups = data["participant"]
    target = data[value]
    cv = GroupKFold(n_splits=5)

    # compare perceived reversibility, the scenario, and participant background
    scenario = ["domain", "state"]
    background_categorical = ["technical_background", "ai_usage_frequency", "ai_agent_use"]
    background_numeric = ["risk_taking", "positive_1", "negative_1", "negative_2", "positive_2"]
    reversibility = ["reversibility"]
    features = data[scenario + background_categorical + background_numeric + reversibility]

    models = {"Median baseline": DummyRegressor(strategy="median")}
    for name, categorical, numeric in [
        ("Reversibility score", [], reversibility),
        ("Scenario", scenario, []),
        ("Scenario and reversibility score", scenario, reversibility),
        ("Scenario, reversibility score, and background",
         scenario + background_categorical, reversibility + background_numeric)
    ]:
        preprocessing = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric)
        ])
        model = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=0)
        models[name] = make_pipeline(preprocessing, model)

    # measure error on unseen participants, then fit the models on all responses
    prediction_results = []
    for name, model in models.items():
        predicted = cross_val_predict(model, features, target, groups=groups, cv=cv)
        prediction_results.append({
            "model": name,
            "n_participants": groups.nunique(),
            "n_responses": len(data),
            "MAE": mean_absolute_error(target, predicted)
        })
        model.fit(features, target)

    prediction_results = pd.DataFrame(prediction_results)
    return prediction_results, models
