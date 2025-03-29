import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Folder containing all CSV log files
DATASET_FOLDER = "/Users/merterol/Desktop/mt-exercise-02/logs"

# Load all CSV files with the expected structure:
#   Epoch, Validation Loss, Validation Perplexity, Phase
log_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]

# Initialize a list to hold the DataFrames
dataframes = []

# Iterate over the log files, read them, and store in a list
for log_file in log_files:
    df = pd.read_csv(
        os.path.join(DATASET_FOLDER, log_file),
        header=None,  # no header in CSV
        names=["Epoch", "Validation Loss", "Validation Perplexity", "Phase"]
    )

    # Extract dataset name from filename
    dataset_name = os.path.splitext(log_file)[0]
    df["dataset"] = dataset_name

    dataframes.append(df)

# Output PDF path
pdf_path = os.path.join(DATASET_FOLDER, "lineplots.pdf")

with PdfPages(pdf_path) as pp:
    for df in dataframes:
        # Split the DataFrame into train/val rows vs. single test row
        df_train = df[df["Phase"] != "test"]
        df_test = df[df["Phase"] == "test"]

        # Create a 2-panel figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- Left plot: Validation Perplexity ---
        # Plot all training/validation epochs as a line
        sns.lineplot(
            ax=axes[0],
            x="Epoch",
            y="Validation Perplexity",
            data=df_train,
            label="Val PPL"
        )
        # Plot the test epoch (if it exists) as a single red dot
        if not df_test.empty:
            sns.scatterplot(
                ax=axes[0],
                x="Epoch",
                y="Validation Perplexity",
                data=df_test,
                color="red",
                marker="o",
                s=100,
                label="Test PPL"
            )
        axes[0].set_title("Validation Perplexity")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Val PPL")
        axes[0].grid(True)

        # --- Right plot: Validation Loss ---
        sns.lineplot(
            ax=axes[1],
            x="Epoch",
            y="Validation Loss",
            data=df_train,
            label="Val Loss"
        )
        if not df_test.empty:
            sns.scatterplot(
                ax=axes[1],
                x="Epoch",
                y="Validation Loss",
                data=df_test,
                color="red",
                marker="o",
                s=100,
                label="Test Loss"
            )
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val Loss")
        axes[1].grid(True)

        # Give a shared title to the figure
        fig.suptitle(f"Validation Metrics over Epochs ({df['dataset'].iloc[0]})")

        # Save current figure to the multipage PDF
        pp.savefig(fig)
        plt.close(fig)

print(f"Saved all plots to {pdf_path}")
