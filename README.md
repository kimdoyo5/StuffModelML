# StuffModelML



## Checklist
- Filter for pitch types for which pitchers have thrown 10+ of each year (due to arm_angles and spin_directions dataset restriction)
- What to do about NA's?

## TO-DO
- [X] Pitch-by-Pitch Arm Angle Estimation
- [X] Pitch-by-Pitch Gyro Angle Estimation
- [X] CleanSavantData.py
- [ ] Trajectory Calculator & Late Movement Calculator (Optional)
- [ ] Model Outline
- [ ] Combine the data into one main dataset so, we need to combine the data that exists plus the data Tom created into one dataset
- [ ] We also need to get the RA9 of all the pitchers and match it to the data
- [ ] Attach the label, so for each data point there should be the last years RA9 and then the label should be this years RA9






## What Each Part Does

### Pitch-by-Pitch Arm Angle Estimation
Currently, BaseballSavant only offers average arm angle for a pitcher, for each year.  
'Pitch-by-Pitch Arm Angle Estimation' estimates pitch-by-pitch arm angle based on release_pos_x and release_pos_z.

Because MLB pitchers are elite athletes, they efficiently rotate in a single plane; their arm swing is in the same plane as torso rotation; wrist and shoulder movement planes are in line with each other.  

ArmAngleEstimator.py assumes the "pivot point" to be roughly 0.5 "lever lengths" from shoulder
- pivot point: point where the wrist/shoulder rotates around
- lever length: distance from shoulder to ball 
    - (think of drawing a straight line from ball to shoulder, then extending 0.5x further into the middle of the body)

AddArmAnglesToSavant.py standardizes that to absolute x-z coordinates (from the relative points in arm_angles_YYYY.csv)

When release_pos_z is higher, or release_pos_x is more gloveside, the estimated_arm_angle is higher.  The opposite is true for lower/armside.

When the pitcher releases the ball later (i.e. more release_extension), both release_pos_x and release_pos_z will drop (closer to pivot point)—so the arm angle doesn't change much. 

### Pitch-by-Pitch Gyro Angle Estimation  
Currently, BaseballSavant only offers average "active spin" for a given pitch type, pitcher, and year.  
'Pitch-by-Pitch Gyro Angle Estimation' estimates pitch-by-pitch gyro angle based on:
- pitch_type_gyro_deg: inferred from active_spin
- release_speed_diff: release_speed change from average (for pitcher's pitch_type's year)
- spin_direction_diff: spin_direction change from average (for pitcher's pitch_type's year)
- spin_rate_diff: spin_rate change from average (for pitcher's pitch_type's year)
- arm_angle_diff: arm_angle change from average (for pitcher's pitch_type's year)
- ssw_tilt: inferred - observed spin axis

The predictor variables were fit to a linear model, categorized by: Fastball, Breaking, Offspeed
(Pitches that were 'none of the above' were noted NA.)
- Fastball: Multiple R^2:  0.3801, Adjusted R^2:  0.3799, p-value: < 2.2e-16
- Breaking: Multiple R^2:  0.6783, Adjusted R^2:  0.6782, p-value: < 2.2e-16
- Offspeed: No correlations
    - Unfortunately, offspeeds showed no relationships even when split into reverse/non-reverse categories
- We found no other models to be superior; this model doesn't appear to overfit.

*Screwballs are assumed to have reverse gyro; Sinkers if their SSW goes in opposite direction; Changeups if they have lower efficiency than the pitcher's highest-efficiency fastball.
All other pitches are assumed to have non-reverse gyro.

### Trajectory Calculator
Incomplete

### Late Movement Calculator
Incomplete





## How to Create Dataset 
#### (Only if something goes wrong)

1. Run CombineCSV.py
2. Run UpdateArmAnglesCSV.py
3. Delete arm_angles_2020-2024.csv (rm "dataset/arm_angles_2020-2024.csv")
4. Run AddArmAnglesToSavant.py
5. (Optional) Run ArmAngleEstimationTest.R
6. Delete savant_data_2020-2024.py (rm "dataset/savant_data_2020-2024.csv")
7. Delete arm_angles_2020-2024_updated.csv (rm "dataset/arm_angles_2020-2024_updated.csv)
8. Run GyroAngleEstimation.R (tests available inside)
9. Delete spin_directions_2020-2024.csv
10. Run CleanSavantData.py

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

#### GyroAngleEstimation.R
Adds 15 columns to savant_data_2020-2024.csv:
- pitch_type_average_arm_angle: Average arm angle for given pitcher, pitch_type, year
- reverse_gyro: boolean
- pitch_group: Fasball/Breaking/Offspeed (or NA)
- pitch_type_release_speed: Average release_speed for given pitcher, pitch_type, year
- pitch_type_spin_rate: Average spin_rate for given pitcher, pitch_type, year
- pitch_type_spin_axis: Average spin_axis for given pitcher, pitch_type, year
- pitch_type_gyro_deg: Average gyro_deg for given pitcher, pitch_type, year
- spin_axis_standardized: LHP's were converted to RHPs
- inferred_spin_axis_standardized: LHP's were converted to RHPs
- release_speed_diff: release_speed - pitch_type_release_speed
- spin_direction_diff: spin_axis - pitch_type_spin_axis
- spin_rate_diff: spin_rate - pitch_type_spin_rate
- arm_angle_diff: estimated_arm_angle - pitch_type_average_arm_angle
- ssw_tilt: inferred - observed spin axis
- estimated_gyro_deg

#### CleanSavantData.py
Removes columns from savant_data_2020-2024.csv:




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
