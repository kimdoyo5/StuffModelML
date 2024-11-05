# StuffModelML

## Checklist
- Filter for pitch types for which pitchers have thrown 10+ of each year (due to arm_angles and spin_directions dataset restriction)

## TO-DO
- Pitch-by-Pitch Spin Profile Estimation
- Trajectory Calculator
- Late Movement Calculator
- CleanSavantData.py


## What Each Part Does

### Pitch-by-Pitch Spin Profile Estimation
Incomplete

### Trajectory Calculator
Incomplete

### Late Movement Calculator
Incomplete


### Pitch-by-Pitch Arm Angle Estimation
Currently, BaseballSavant only offers average arm angle for a pitcher, for each year.  
This estimates pitch-by-pitch arm angle based on release_pos_x and release_pos_z.

Because MLB pitchers are elite athletes, they efficiently rotate in a single plane; their arm swing is in the same plane as torso rotation; wrist and shoulder movement planes are in line with each other.  

ArmAngleEstimator.py assumes the "pivot point" to be roughly 0.5 "lever lengths" from shoulder
- pivot point: point where the wrist/shoulder rotates around
- lever length: distance from shoulder to ball 
    - (think of drawing a straight line from ball to shoulder, then extending 0.5x further into the middle of the body)

AddArmAnglesToSavant.py standardizes that to absolute x-z coordinates (from the relative points in arm_angles_YYYY.csv)

When release_pos_z is higher, or release_pos_x is more gloveside, the estimated_arm_angle is higher.  The opposite is true for lower/armside.

When the pitcher releases the ball later (i.e. more release_extension), both release_pos_x and release_pos_z will drop (closer to pivot point)—so the arm angle doesn't change much. 




## How to Create Dataset 
#### (Only if something goes wrong)

1. Run CombineCSV.py
2. Run UpdateArmAnglesCSV.py
3. Delete arm_angles_2020-2024.csv (rm "dataset/arm_angles_2020-2024.csv")
4. Run AddArmAnglesToSavant.py
5. Delete savant_data_2020-2024.py (rm "dataset/savant_data_2020-2024.csv")
6. Delete arm_angles_2020-2024_updated.csv (rm "dataset/arm_angles_2020-2024_updated.csv)

### Explanation
#### CombineCSV.py 
- arm_angles_2020.csv, ..., arm_angles_2024.csv TO arm_angles_2020-2024.csv
- savant_data_2020.csv, ..., savant_data_2024.csv TO savant_data_2020-2024.csv
- spin_directions_2020.csv, ..., spin_directions_2024.csv TO spin_directions_2020-2024.csv

#### UpdateArmAnglesCSV.py 
Adds 4 columns to arm_angles_2020-2024.csv:
- precise_average_arm_angle: ball_angle without rounding
- lever_length: (euclidean) distance from shoulder to ball (ft)
- pivot_point_x, pivot_point_y: coordinate where the body is "assumed" to rotate around

#### AddArmAnglesToSavant.py 
Adds 3 columns to savant_data_2020-2024.csv:
- year
- estimated_arm_angle: pitch-by-pitch
- precise_average_arm_angle


## Dataset Explanation

### Raw_data
- savant_data_YYYY: has all the pitch_by_pitch data
- arm_angles_YYYY: has all the arm angle data (per year, per pitcher)
- spin_directions_YYYY: has all the spin direction/efficiency data (per year, per pitch type, per pitcher) 
    - (savant_data has direction data, but not efficiency)

#### arm_angles_YYYY 
Contains arm angle data
- pitcher_name 
- year 
- pitch_hand: L/R
- n_pitches 
- team_id
- ball_angle: average arm angle
- relative_release_ball_x: average horizontal release position (relative to center of mass)
- release_ball_z: average release height
- relative_shoulder_x: average horizontal shoulder position (relative to center of mass) 
- shoulder_z: average shoulder height
