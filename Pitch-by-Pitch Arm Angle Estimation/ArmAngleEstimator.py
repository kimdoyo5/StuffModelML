import pandas as pd
import numpy as np
import math

# append a column in arm_angles_2020 
# name of column should be precise_average_arm_angle
# Apply the function to each row in the DataFrame and create the new column
def precise_average_arm_angle(relative_shoulder_x, shoulder_z, relative_release_ball_x, ball_z, pitch_hand):
    z_diff = ball_z - shoulder_z
    x_diff = relative_release_ball_x - relative_shoulder_x
    angle_rad = math.atan(z_diff/x_diff)
    if pitch_hand == "R":
        return -math.degrees(angle_rad)
    return math.degrees(angle_rad) 

# Add lever length
# Lever length is simply distance between shoulder and ball
def calc_lever_length(relative_shoulder_x, shoulder_z, relative_release_ball_x, ball_z):
    z_diff = ball_z - shoulder_z
    x_diff = relative_release_ball_x - relative_shoulder_x
    vector = [x_diff, z_diff]
    
    # Calculate the norm (Euclidean distance)
    lever = np.linalg.norm(vector)
    return lever

# Add pivot point
# Pivot point is the point in body that doesn't change regardless of release point change
# Because it's the point where the body rotates around
# Assume that arm angle is in-line with the rotation angle
# Assume pivot point is 0.5x lever_length
# I.e. draw a straight line from ball to shoulder, and pivot point is 0.5x lever length further
# Apply the function and store the results in a new DataFrame
def calc_pivot_point(precise_average_arm_angle, lever_length, relative_shoulder_x, shoulder_z, pitch_hand):
    # Convert the angle from degrees to radians for trigonometric calculations
    angle_rad = math.radians(precise_average_arm_angle)
    
    # Calculate the pivot distance (0.5 times lever length)
    pivot_distance = 0.5 * lever_length
    
    # Calculate the X and Z offsets based on the angle
    x_offset = pivot_distance * math.cos(angle_rad)
    z_offset = pivot_distance * math.sin(angle_rad)
    
    # Calculate the pivot point coordinates
    pivot_x = relative_shoulder_x + x_offset if pitch_hand == "R" else relative_shoulder_x - x_offset
    pivot_z = shoulder_z - z_offset
    
    return pivot_x, pivot_z

# Pivot point above is defined relative to Center of Mass
# Finding its exact coordinate
def pivot_point_absolute(pivot_point_x, pivot_point_z, relative_release_ball_x, release_ball_z, avg_release_pos_x, avg_release_pos_z):
    x_pos_diff = relative_release_ball_x - avg_release_pos_x
    z_pos_diff = release_ball_z - avg_release_pos_z
    abs_pivot_x = pivot_point_x - x_pos_diff
    abs_pivot_z = pivot_point_z - z_pos_diff
    return abs_pivot_x, abs_pivot_z

# With absolute pivot points, we can estimate arm angle for each individual pitch
def individual_arm_angle(abs_pivot_x, abs_pivot_z, release_pos_x, release_pos_z, pitch_hand):
    return precise_average_arm_angle(abs_pivot_x, abs_pivot_z, release_pos_x, release_pos_z, pitch_hand)



# Test
if __name__ == "__main__":
    
    arm_angles_2020 = pd.read_csv("raw_data/arm_angles_2020.csv")
    print(arm_angles_2020.head())

    arm_angles_2020['precise_average_arm_angle'] = arm_angles_2020.apply(
        lambda row: precise_average_arm_angle(row['relative_shoulder_x'], row['shoulder_z'], 
                                    row['relative_release_ball_x'], row['release_ball_z'],
                                    row['pitch_hand']), 
        axis=1
    )

    arm_angles_2020['lever_length'] = arm_angles_2020.apply(
        lambda row: calc_lever_length(row['relative_shoulder_x'], row['shoulder_z'], 
                                    row['relative_release_ball_x'], row['release_ball_z']), 
        axis=1
    )

    pivot_points = arm_angles_2020.apply(
        lambda row: calc_pivot_point(row['precise_average_arm_angle'], row['lever_length'],
                                    row['relative_shoulder_x'], row['shoulder_z'],
                                    row['pitch_hand']),
        axis=1
    )
    # Convert the resulting Series of tuples into a DataFrame
    pivot_points_df = pd.DataFrame(pivot_points.tolist(), columns=['pivot_point_x', 'pivot_point_z'])
    # Concatenate the new columns to the original DataFrame
    arm_angles_2020 = pd.concat([arm_angles_2020, pivot_points_df], axis=1)

    # print
    sample_data = arm_angles_2020[arm_angles_2020['pitcher_name'].isin(
        ["Rogers, Tyler", "Hill, Tim", "Buehler, Walker", "Heaney, Andrew"])][
            ['pitcher_name', 'ball_angle', 'precise_average_arm_angle', 'lever_length', 
            'relative_shoulder_x', 'shoulder_z', 'pivot_point_x', 'pivot_point_z']]
    print(sample_data)
