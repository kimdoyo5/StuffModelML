# Update the arm_angles_YYYY files
# Adding precise_average_arm_angle, lever_length, and pivot point

import pandas as pd
from ArmAngleEstimator import precise_average_arm_angle, calc_lever_length, calc_pivot_point

# Load the data
df = pd.read_csv("raw_data/arm_angles_2020-2024.csv")

# Calculate and append the new columns
df['precise_average_arm_angle'] = df.apply(
    lambda row: precise_average_arm_angle(row['relative_shoulder_x'], row['shoulder_z'], 
                                    row['relative_release_ball_x'], row['release_ball_z'],
                                    row['pitch_hand']), 
    axis=1
)

df['lever_length'] = df.apply(
    lambda row: calc_lever_length(row['relative_shoulder_x'], row['shoulder_z'], 
                                    row['relative_release_ball_x'], row['release_ball_z']), 
    axis=1
)

pivot_points = df.apply(
    lambda row: calc_pivot_point(row['precise_average_arm_angle'], row['lever_length'],
                                    row['relative_shoulder_x'], row['shoulder_z'],
                                    row['pitch_hand']),
    axis=1
)
# Convert the resulting Series of tuples into a DataFrame
pivot_points_df = pd.DataFrame(pivot_points.tolist(), columns=['pivot_point_x', 'pivot_point_z'])
# Concatenate the new columns to the original DataFrame
df = pd.concat([df, pivot_points_df], axis=1)

# Write the updated DataFrame to the new CSV file
df.to_csv("dataset/arm_angles_2020-2024_updated.csv", index=False)

print("Updated CSV files have been written with '_updated' suffix.")
