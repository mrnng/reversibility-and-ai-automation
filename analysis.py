import argparse
import os

import pandas as pd
from utils import *

# keep the original data directory, or use responses.csv beside this script
script_directory = os.path.dirname(os.path.abspath(__file__))
data_file = "./data/responses.csv"
if not os.path.isfile(data_file):
    data_file = os.path.join(script_directory, "responses.csv")

parser = argparse.ArgumentParser(description="Analyze the reversibility survey responses.")
parser.add_argument("responses", nargs="?", default=data_file, help="path to the responses CSV")
data_file = parser.parse_args().responses

results_directory = os.path.join(script_directory, "results")
os.makedirs(results_directory, exist_ok=True)

# constants
OUTCOMES = ["reversibility", "willingness", "risk", "bother"]

# load dataset
df = pd.read_csv(data_file)

df = prepare_repeated_data(df)

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

all_page = []
all_friedman = []

# perform Page rank tests
for outcome in ["reversibility", "willingness"]:
    page_results = perform_page_trend_test([df_domain_1, df_domain_2], outcome, direction="increasing")
    all_page.append(page_results.assign(outcome=outcome))
    print(f"Page rank test results for {outcome}:")
    print(page_results)
    print()
for outcome in ["risk", "bother"]:
    page_results = perform_page_trend_test([df_domain_1, df_domain_2], outcome, direction="decreasing")
    all_page.append(page_results.assign(outcome=outcome))
    print(f"Page rank test results for {outcome}:")
    print(page_results)
    print()

# perform Friedman tests
for outcome in OUTCOMES:
    friedman_results = perform_friedman_test([df_domain_1, df_domain_2], outcome)
    all_friedman.append(friedman_results.assign(outcome=outcome))
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


# calculate the participant counts and descriptive means reported in the paper
participant_results = participant_summary(df)
print("Participant summary:")
print(participant_results.to_string(index=False))
print()
participant_results.to_csv(os.path.join(results_directory, "participants.csv"), index=False)

sample_results = pd.DataFrame([{
    "participants": df["participant"].nunique(),
    "observations": len(df),
    "states": df["state"].nunique(),
    "domains": df["domain"].nunique()
}])
print("Sample size:")
print(sample_results)
print()
sample_results.to_csv(os.path.join(results_directory, "sample_size.csv"), index=False)

# each participant has both domains, so these are equal-domain means
pooled_means = df.groupby("state", observed=True)[OUTCOMES].mean()
domain_means = df.groupby(["domain", "state"], observed=True)[OUTCOMES].agg(["mean", "std", "count"])
print("Pooled means by state:")
print(pooled_means)
print()
pooled_means.to_csv(os.path.join(results_directory, "pooled_means.csv"))
domain_means.to_csv(os.path.join(results_directory, "descriptives.csv"))
pd.concat(all_pairwise, ignore_index=True).to_csv(os.path.join(results_directory, "rank_pairwise.csv"), index=False)

# fit the models, print their summaries, and collect the estimates
all_model_results = []
all_contrasts = []
all_interactions = []

for outcome in OUTCOMES:
    model, fixed_effects, model_results = perform_mixed_effects_model(df, outcome)
    contrasts = mixed_effects_contrasts(df, model, outcome)
    interaction = domain_state_interaction(df, outcome)

    all_model_results.append(model_results)
    all_contrasts.append(contrasts)
    all_interactions.append(interaction)

    print(f"Mixed-effects model for {outcome}:")
    print(model.summary())
    print(model_results.to_string(index=False))
    print()

    fixed_effects.to_csv(os.path.join(results_directory, f"{outcome}_fixed_effects.csv"), index=False)
    with open(os.path.join(results_directory, f"{outcome}_mixed_model.txt"), "w") as file:
        file.write(str(model.summary()))
        file.write("\nWarnings: " + model_results["warnings"].iloc[0])

# collect all contrasts before applying the stated Holm correction families
mixed_effects_results = pd.concat(all_model_results, ignore_index=True)
contrast_results = pd.concat(all_contrasts, ignore_index=True)
contrast_results = correct_mixed_effects_contrasts(contrast_results)
interaction_results = pd.DataFrame(all_interactions)
interaction_results["p_holm"] = holm_correction(interaction_results["p"].to_numpy())

print("Mixed-effects marginal contrasts:")
print(contrast_results.to_string(index=False))
print()
print("State-by-domain interaction tests (participant-level Hotelling tests):")
print(interaction_results.to_string(index=False))
print()

mixed_effects_results.to_csv(os.path.join(results_directory, "mixed_effects_models.csv"), index=False)
contrast_results.to_csv(os.path.join(results_directory, "marginal_contrasts.csv"), index=False)
interaction_results.to_csv(os.path.join(results_directory, "domain_state_interactions.csv"), index=False)

# display the same point estimates and pointwise bootstrap intervals as Table 2
# the p values above are participant-contrast t tests, not model Wald p values
adjacent = contrast_results["comparison"].str.contains("->")
table_results = contrast_results.loc[adjacent].copy()
table_results["estimate_and_CI"] = table_results.apply(
    lambda row: f'{row["estimate"]:+.2f} [{row["CI_low"]:.2f}, {row["CI_high"]:.2f}]', axis=1
)
table_results = table_results.pivot(index="outcome", columns="comparison", values="estimate_and_CI")
table_results = table_results.reindex(
    index=["willingness", "risk", "bother", "reversibility"],
    columns=["S0->S1", "S1->S2-H", "S2-H->S2-A"]
)
print("Table 2 (pointwise bootstrap confidence intervals):")
print(table_results.to_string())
print()
table_results.to_csv(os.path.join(results_directory, "paper_table_2.csv"))

# repeat the models and contrasts without the flagged rating profiles
# everyone remains in the main analysis; identical ratings may be legitimate
invariant, duplicate_later, excluded = rating_quality_flags(df)
quality_results = pd.DataFrame({
    "participant": excluded,
    "invariant": [participant in invariant for participant in excluded],
    "duplicate_later": [participant in duplicate_later for participant in excluded]
})
print("Profiles excluded only in the sensitivity analysis:")
print(quality_results)
print()
quality_results.to_csv(os.path.join(results_directory, "quality_flags.csv"), index=False)

df_sensitivity = df[~df["participant"].isin(excluded)].copy()
all_sensitivity = []
all_sensitivity_models = []
all_sensitivity_fixed_effects = []

for outcome in OUTCOMES:
    model, fixed_effects, model_results = perform_mixed_effects_model(df_sensitivity, outcome)
    contrasts = mixed_effects_contrasts(df_sensitivity, model, outcome)
    all_sensitivity.append(contrasts)
    all_sensitivity_models.append(model_results)
    all_sensitivity_fixed_effects.append(fixed_effects.assign(outcome=outcome))

    with open(os.path.join(results_directory, f"{outcome}_sensitivity_model.txt"), "w") as file:
        file.write(str(model.summary()))
        file.write("\nWarnings: " + model_results["warnings"].iloc[0])

sensitivity_results = pd.concat(all_sensitivity, ignore_index=True)
sensitivity_results = correct_mixed_effects_contrasts(sensitivity_results)
sensitivity_models = pd.concat(all_sensitivity_models, ignore_index=True)
print("Sensitivity contrasts:")
print(sensitivity_results.to_string(index=False))
print()
sensitivity_results.to_csv(os.path.join(results_directory, "sensitivity_contrasts.csv"), index=False)
sensitivity_models.to_csv(os.path.join(results_directory, "sensitivity_models.csv"), index=False)
sensitivity_fixed_effects = pd.concat(all_sensitivity_fixed_effects, ignore_index=True)
sensitivity_fixed_effects.to_csv(os.path.join(results_directory, "sensitivity_fixed_effects.csv"), index=False)

# free-text answers are not in responses.csv, so qualitative counts cannot be
# calculated here; do not insert manually copied qualitative counts as results

# predict the ratings using reversibility and other survey information
df_prediction = pd.concat([
    df_domain_1.assign(domain="file"),
    df_domain_2.assign(domain="messaging")
], ignore_index=True)

prediction_models = {}
all_prediction = []
prediction_outcomes = ["willingness", "risk", "bother"]
for outcome in prediction_outcomes:
    prediction_results, models = perform_prediction(df_prediction, outcome)
    prediction_models[outcome] = models
    all_prediction.append(prediction_results.assign(outcome=outcome))
    print(f"Prediction results for {outcome}:")
    print(prediction_results.round(3).to_string(index=False))
    print()

# save the results for all outcomes
for name, tables in [
    ("page_rank_results", all_page),
    ("friedman_results", all_friedman),
    ("wilcoxon_results", all_pairwise),
    ("prediction_results", all_prediction)
]:
    results = pd.concat(tables, ignore_index=True)
    results.to_csv(os.path.join(results_directory, f"{name}.csv"), index=False)

# show the mean absolute error for each feature combination side by side
prediction_results = pd.concat(all_prediction, ignore_index=True)
prediction_table = prediction_results.pivot(index="model", columns="outcome", values="MAE")
prediction_table = prediction_table.reindex(
    index=prediction_results["model"].drop_duplicates(), columns=prediction_outcomes
)
print("Prediction mean absolute error by feature combination:")
print(prediction_table.round(3).to_string())
print()
prediction_table.to_csv(os.path.join(results_directory, "prediction_mae_by_feature_set.csv"))
