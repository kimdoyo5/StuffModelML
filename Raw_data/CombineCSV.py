import pandas as pd

# List of updated CSV file paths
spin_directions = [
    "raw_data/spin_directions_2020.csv",
    "raw_data/spin_directions_2021.csv",
    "raw_data/spin_directions_2022.csv",
    "raw_data/spin_directions_2023.csv",
    "raw_data/spin_directions_2024.csv"
]

# Read each CSV file and store in a list of DataFrames
dataframes = [pd.read_csv(file) for file in spin_directions]

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Export the combined DataFrame to a single CSV file
combined_df.to_csv("spin_directions_2020-2024.csv", index=False)

print("All updated files have been combined and saved as 'spin_directions.csv'.")
