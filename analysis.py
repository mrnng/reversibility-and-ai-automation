import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from utils import *

# load dataset
df = pd.read_csv("./data/responses.csv")

# split by domain
df_domain_1 = df[df["task"].between(1, 4)].copy()
df_domain_2 = df[df["task"].between(5, 8)].copy()

# map each task to its state
state_map = {
    1: "S0", 2: "S1", 3: "S2", 4:"S3",
    5: "S0", 6: "S1", 7: "S2", 8:"S3",
}

df_domain_1["state"] = df_domain_1["task"].map(state_map)
df_domain_2["state"] = df_domain_2["task"].map(state_map)

reversibility_page_trend_results = perform_page_trend_test([df_domain_1, df_domain_2], "reversibility")
print(reversibility_page_trend_results)
