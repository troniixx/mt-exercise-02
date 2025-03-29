import os
import pandas as pd

DATASET_FOLDER = "/Users/merterol/Desktop/mt-exercise-02/logs"

log_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]

dataframes = []
for log_file in log_files:
    # Each file has columns (no header):
    # Epoch, Validation Loss, Validation Perplexity, Phase
    df = pd.read_csv(
        os.path.join(DATASET_FOLDER, log_file),
        header=None,
        names=["Epoch", "Validation Loss", "Validation Perplexity", "Phase"]
    )
    # Extract a dataset name (e.g. "dropout_0.1" from the filename)
    dataset_name = os.path.splitext(log_file)[0]
    df["dataset"] = dataset_name
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df = combined_df[combined_df["Phase"] != "test"].copy()
table_loss = combined_df.pivot(index="Epoch", columns="dataset", values="Validation Loss")
table_ppl = combined_df.pivot(index="Epoch", columns="dataset", values="Validation Perplexity")

# 5) Now you have two DataFrames in wide format:
print("=== Validation Loss Table ===")
print(table_loss)

print("\n=== Validation Perplexity Table ===")
print(table_ppl)