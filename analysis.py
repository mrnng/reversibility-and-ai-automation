import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from utils import *

# constants
OUTCOMES = ["reversibility", "willingness", "risk", "bother"]

# load dataset
df = pd.read_csv("./data/responses.csv")

# split by domain
df_domain_1 = df[df["task"].between(1, 4)].copy()
df_domain_2 = df[df["task"].between(5, 8)].copy()

# map each task to its state
state_map = {
    1: "S0", 2: "S1", 3: "S2-H", 4:"S2-A",
    5: "S0", 6: "S1", 7: "S2-H", 8:"S2-A",
}

df_domain_1["state"] = df_domain_1["task"].map(state_map)
df_domain_2["state"] = df_domain_2["task"].map(state_map)

# perform Page rank tests
for outcome in ["reversibility", "willingness"]:
    page_results = perform_page_trend_test([df_domain_1, df_domain_2], outcome, direction="increasing")
    print(f"Page rank test results for {outcome}:")
    print(page_results)
    print()
for outcome in ["risk", "bother"]:
    page_results = perform_page_trend_test([df_domain_1, df_domain_2], outcome, direction="decreasing")
    print(f"Page rank test results for {outcome}:")
    print(page_results)
    print()

# perform Friedman tests
for outcome in OUTCOMES:
    friedman_results = perform_friedman_test([df_domain_1, df_domain_2], outcome)
    print(f"Friedman test results for {outcome}:")
    print(friedman_results)
    print()

# perform Wilcoxon tests
all_pairwise = []
for outcome in OUTCOMES:
    pairwise_results = perform_pairwise_tests([df_domain_1, df_domain_2], outcome)
    pairwise_results.insert(0, "outcome", outcome)
    all_pairwise.append(pairwise_results)
    print(f"Adjacent Wilcoxon tests for {outcome}:")
    print(pairwise_results)
    print()
