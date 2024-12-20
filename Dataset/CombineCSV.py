import pandas as pd

########## Combine savant_data_YYYY.csv ##################################
savant_data = [
    "Dataset/Raw_data/savant_data_2020.csv",
    "Dataset/Raw_data/savant_data_2021.csv",
    "Dataset/Raw_data/savant_data_2022.csv",
    "Dataset/Raw_data/savant_data_2023.csv",
    "Dataset/Raw_data/savant_data_2024.csv"
]

# Read each CSV file and store in a list of DataFrames
dataframes = [pd.read_csv(file) for file in savant_data]

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Export the combined DataFrame to a single CSV file
combined_df.to_csv("Dataset/savant_data_2020-2024.csv", index=False)

print("All updated files have been combined and saved as 'savant_data_2020-2024.csv'.")


########## Combine arm_angles_YYYY.csv ##################################
arm_angles = [
    "Dataset/Raw_data/arm_angles_2020.csv",
    "Dataset/Raw_data/arm_angles_2021.csv",
    "Dataset/Raw_data/arm_angles_2022.csv",
    "Dataset/Raw_data/arm_angles_2023.csv",
    "Dataset/Raw_data/arm_angles_2024.csv"
]

dataframes = [pd.read_csv(file) for file in arm_angles]
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv("Dataset/arm_angles_2020-2024.csv", index=False)

print("All updated files have been combined and saved as 'arm_angles_2020-2024.csv'.")


########## Combine spin_directions_YYYY.csv ##################################
spin_directions = [
    ("Dataset/Raw_data/spin_directions_2020.csv", 2020),
    ("Dataset/Raw_data/spin_directions_2021.csv", 2021),
    ("Dataset/Raw_data/spin_directions_2022.csv", 2022),
    ("Dataset/Raw_data/spin_directions_2023.csv", 2023),
    ("Dataset/Raw_data/spin_directions_2024.csv", 2024)
]

# Load each file, add the 'year' column, and store in a list
dataframes = []
for file, year in spin_directions:
    df = pd.read_csv(file)
    df['year'] = year  # Add year column
    dataframes.append(df)
    
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv("Dataset/spin_directions_2020-2024.csv", index=False)

print("All updated files have been combined and saved as 'spin_directions_2020-2024.csv'.")