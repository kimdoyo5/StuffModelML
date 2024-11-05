import pandas as pd
from ArmAngleEstimator import pivot_point_absolute, individual_arm_angle

# Load data
arm_angles_updated = pd.read_csv("dataset/arm_angles_2020-2024_updated.csv")
print("arm_angles_updated loaded")
savant_data = pd.read_csv("dataset/savant_data_2020-2024.csv")
print("savant_data loaded")


########## Step 1. Find pivot points #########################################
print("Step 1. Find pivot points")

# Calculate avg_release_pos_x and avg_release_pos_z for each pitcher-year in savant_data
savant_data['year'] = pd.to_datetime(savant_data['game_date']).dt.year  # matching year
avg_release_pos = savant_data.groupby(['pitcher', 'year'])[['release_pos_x', 'release_pos_z']].mean().reset_index()
avg_release_pos.rename(columns={'release_pos_x': 'avg_release_pos_x', 'release_pos_z': 'avg_release_pos_z'}, inplace=True)


# Merge avg_release_pos with arm_angles_updated to get avg_release_pos_x and avg_release_pos_z for each pitcher-year
arm_angles_updated = arm_angles_updated.merge(avg_release_pos, on=['pitcher', 'year'], how='left')


# Calculate absolute pivot points for each pitcher's year using pivot_point_absolute
arm_angles_updated[['abs_pivot_x', 'abs_pivot_z']] = arm_angles_updated.apply(
    lambda row: pivot_point_absolute(
        row['pivot_point_x'], row['pivot_point_z'],
        row['relative_release_ball_x'], row['release_ball_z'],
        row['avg_release_pos_x'], row['avg_release_pos_z']
    ),
    axis=1, result_type='expand'
)

# Test
print("Test in progress...")
sample_data = arm_angles_updated[arm_angles_updated['pitcher_name'].isin(
    ["Rogers, Tyler", "Hill, Tim", "Buehler, Walker", "Heaney, Andrew"])][
        ['pitcher_name', 'year', 'ball_angle', 'precise_average_arm_angle', 'lever_length', 
        'relative_shoulder_x', 'shoulder_z', 'pivot_point_x', 'pivot_point_z']]
print(sample_data)


########## Step 2. Estimate Arm Angle for each pitch #########################
print("Step 2. Estimate Arm Angle for each pitch")



# Merge updated arm_angles data with savant_data to match each pitch with the absolute pivot points
savant_data = savant_data.merge(
    arm_angles_updated[['pitcher', 'year', 'abs_pivot_x', 'abs_pivot_z', 'precise_average_arm_angle'
                        # DEBUGGING: 'relative_shoulder_x', 'shoulder_z', 'relative_release_ball_x','release_ball_z',
                        #            'avg_release_pos_x', 'avg_release_pos_z', 'pivot_point_x', 'pivot_point_z'
                        ]],
    on=['pitcher', 'year'], how='left'
)

# Calculate arm angle for each pitch using individual_arm_angle
print("Calculating arm angle for each pitch...")
savant_data['estimated_arm_angle'] = savant_data.apply(
    lambda row: individual_arm_angle(
        row['release_pos_x'], row['release_pos_z'],
        row['abs_pivot_x'], row['abs_pivot_z'],
        row['p_throws']
    ),
    axis=1
)

# DEBUGGING: Calculate estimated arm angle for average numbers —— confirmed equal to precise_average_arm_angle
# savant_data['average_estimated_arm_angle'] = savant_data.apply(
#     lambda row: individual_arm_angle(
#         row['avg_release_pos_x'], row['avg_release_pos_z'],
#         row['abs_pivot_x'], row['abs_pivot_z'],
#         row['p_throws']
#     ),
#     axis=1
# )

# DEBUGGING: pivot-based arm angle (pre-abs_pivot) —— confirmed equal to precise_average_arm_angle
# savant_data['pivot_based_arm_angle'] = savant_data.apply(
#     lambda row: individual_arm_angle(
#         row['relative_release_ball_x'], row['release_ball_z'],
#         row['pivot_point_x'], row['pivot_point_z'],
#         row['p_throws']
#     ),
#     axis=1
# )

# Test
print("Test in progress...")
sample_data = savant_data[savant_data['player_name'].isin(
    ["Rogers, Tyler", "Hill, Tim", "Buehler, Walker", "Heaney, Andrew"])][
        ['player_name', 'abs_pivot_x', 'abs_pivot_z', 'release_pos_x', 'release_pos_z', 'estimated_arm_angle', 'precise_average_arm_angle'
        # Debugging: 'pivot_point_x', 'pivot_point_z', 'relative_release_ball_x', 'release_ball_z', 'relative_shoulder_x', 'shoulder_z', 
        #            'average_estimated_arm_angle', 'pivot_based_arm_angle'
        ]]
print(sample_data)


########## Step 3. Save updated data #########################################
print("Saving...")

# Remove 'abs_pivot_x' and 'abs_pivot_z' columns before saving
savant_data = savant_data.drop(columns=['abs_pivot_x', 'abs_pivot_z'])

# Save the updated savant data with arm angle estimates
savant_data.to_csv("dataset/savant_data_2020-2024_updated.csv", index=False)

print("Update completed. Saved to 'dataset/savant_data_2020-2024_updated.csv'")
