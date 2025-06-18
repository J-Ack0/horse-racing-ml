import pandas as pd
import glob
import os

# Set the path to your directory containing the batch CSV files
csv_folder_path = "Batch"
output_file_path = "merged_horses.csv"

# Use glob to find all relevant CSV files
csv_files = glob.glob(os.path.join(csv_folder_path, "batch_horses_batch_*.csv"))

# Load and concatenate all CSV files
all_dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    all_dfs.append(df)

# Combine and drop duplicates
merged_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

# Save to output
merged_df.to_csv(output_file_path, index=False)

print(f"Merged {len(csv_files)} files into: {output_file_path}")
