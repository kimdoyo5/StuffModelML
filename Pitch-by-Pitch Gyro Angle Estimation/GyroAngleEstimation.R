##################################################################################################
# Step 1: Load the necessary libraries and read in the datasets.
library(ggplot2)
library(dplyr)
library(mgcv)


# The 'spin' dataset contains spin direction data for pitches from 2020 to 2024.
# The 'savant_data' dataset contains detailed pitch data from Statcast.

# Read in data
spin <- read.csv("Dataset/spin_directions_2020-2024.csv")
head(spin)

savant_data <- read.csv("Dataset/savant_data_2020-2024_updated.csv") 
head(savant_data)

# Get unique pitch types from both datasets
print(unique(spin$api_pitch_type))
print(unique(savant_data$pitch_type))



##################################################################################################
# Step 2: Data Preprocessing
# - Create a new variable 'player_name' by combining last and first names.
# - Categorize pitches into 'Fastball', 'Breaking', 'Offspeed', or 'Other' based on 'api_pitch_type'.
# - Select relevant columns for analysis.
# - Rearrange columns to move 'player_name' after 'year'.
# - Rename 'api_pitch_type' to 'pitch_type'.

# Categorize into 3 categories
# Then, comparing pitches, by the same pitcher, in different categories, different year
# (Say, pitcher A's FF changed between 2020 and 2023, linear model against gyro_deg)
# (Say, pitcher B's FF is different from CT, linear model against gyro_deg)

spin1 <- spin |>
  mutate(
    player_name = last_name..first_name,
    pitch_group = case_when(
      api_pitch_type %in% c("FF","SI") ~ "Fastball",
      api_pitch_type %in% c("SL","ST","CU","SV") ~ "Breaking",
      api_pitch_type %in% c("CH","FS") ~ "Offspeed",
      TRUE ~ "Other"
    )
  ) |>
  select(
    year, player_name, player_id, pitch_hand, api_pitch_type, pitch_group, 
    n_pitches, release_speed, spin_rate, movement_inches, alan_active_spin_pct,
    active_spin, hawkeye_measured, movement_inferred
  ) |>
  relocate(player_name, .after = 1) |>
  rename(pitch_type = api_pitch_type)

head(spin1)

########## Calculate average arm angles and merge with 'spin1' dataset ########## 
# - From 'savant_data', select relevant columns and calculate the average 'estimated_arm_angle' for each player, year, and pitch type.
# - Merge 'spin1' with the calculated average arm angles ('savant_used').

# Calculate arm angles
savant_used <- savant_data |>
  select(player_name, year, pitch_type, estimated_arm_angle) |>
  group_by(player_name, year, pitch_type) |>
  summarise(avg_estimated_arm_angle = mean(estimated_arm_angle, na.rm = TRUE))

head(savant_used)

spin2 <- spin1 |>
  inner_join(savant_used, by = c("player_name","pitch_type","year"))

head(spin2)

########## Prepare data for comparing pitches of the same group by the same pitcher across different years
# - Perform a self-join on 'spin2' based on 'pitch_group' and 'player_id'.
# - Use suffixes '_left' and '_right' to distinguish between the two datasets in the join.
# - Filter out rows where 'year_left' is equal to 'year_right' to ensure we are comparing different years.
# - Select relevant columns for analysis.
# - Rename 'player_name_left' to 'player_name' and 'pitch_hand_left' to 'pitch_hand' for clarity.

# Two of the same pitch group, same pitcher, different year
spin_joined <- spin2 %>%
  inner_join(spin2, by = c("pitch_group", "player_id"), suffix = c("_left", "_right")) %>%
  filter(year_left != year_right) %>%
  select(
    player_name_left, player_id, pitch_hand_left, pitch_group, 
    pitch_type_left, pitch_type_right,
    year_left, year_right,
    n_pitches_left, n_pitches_right,
    release_speed_left, release_speed_right,
    spin_rate_left, spin_rate_right,
    movement_inches_left, movement_inches_right,
    alan_active_spin_pct_left, alan_active_spin_pct_right,
    active_spin_left, active_spin_right,
    hawkeye_measured_left, hawkeye_measured_right,
    movement_inferred_left, movement_inferred_right,
    avg_estimated_arm_angle_left, avg_estimated_arm_angle_right
  ) %>%
  rename(
    player_name = player_name_left,
    pitch_hand = pitch_hand_left
  )

head(spin_joined, 50)

########## Categorize pitches as reverse-gyro or not
# - Assume only Changeups ('CH') might be reverse-gyro.
# - For each pitcher and year, compare the active spin of the 'CH' to the highest active spin of their fastballs ('FF', 'CT', 'SI').
# - If the active spin of the 'CH' is less than the highest active spin of the fastballs, classify it as reverse-gyro.
# - Join this information back to 'spin_joined' to create 'spin_joined0', adding 'reverse_gyro_left' and 'reverse_gyro_right' indicators.

# Categorize whether reverse-gyro or not
# Assume no pitch is reverse-gyro except CH
# Decision: If more gyro than highest-efficiency fastball (of 3), reverse-gyro 
# (Inaccurate with ~100% efficiency pitches, but then doesn't matter much)

reverse_gyro <- spin1 |>
  inner_join(spin1, by = c("year", "player_id"), suffix = c("_left", "_right")) |>
  filter(
    pitch_type_right == "CH",
    pitch_type_left %in% c("FF", "CT", "SI")
  ) |>
  group_by(player_id, year, pitch_type_right) |>
  summarise(
    highest_FB_active_spin = max(active_spin_left, na.rm = TRUE),
    active_spin_right = unique(active_spin_right)
  ) |>
  select(year, player_id, pitch_type_right, highest_FB_active_spin, active_spin_right) |>
  mutate(reverse_gyro = if_else(highest_FB_active_spin > active_spin_right, 1, 0)) |>
  rename(year_right = year)

head(reverse_gyro)

spin_joined0 <- spin_joined |>
  left_join(reverse_gyro, by = c("year_right", "player_id", "pitch_type_right", "active_spin_right")) |>
  mutate(reverse_gyro = if_else(pitch_type_right != "CH", 0, reverse_gyro)) |>
  rename(reverse_gyro_right = reverse_gyro)

reverse_gyro <- reverse_gyro |> # Left
  rename(
    year_left = year_right,
    pitch_type_left = pitch_type_right,
    active_spin_left = active_spin_right
  )

spin_joined0 <- spin_joined0 |>
  left_join(reverse_gyro, by = c("year_left", "player_id", "pitch_type_left", "active_spin_left"), suffix = c("_left", "_right")) |>
  mutate(reverse_gyro = if_else(pitch_type_left != "CH", 0, reverse_gyro)) |>
  rename(reverse_gyro_left = reverse_gyro)

head(spin_joined0)

########## Standardize spin direction to right-handed pitchers (RHP)
# - For left-handed pitchers (LHP), adjust spin direction measurements by subtracting from 360 degrees.
# - This standardization allows for consistent comparison regardless of pitcher's handedness.
# - Remove the original spin direction and movement columns, as well as 'pitch_hand' column.

# Standardize spin direction (for handedness) to RHP
spin_joined1 <- spin_joined0 %>%
  mutate(
    hawkeye_measured_left_standardized = if_else(pitch_hand == "L", 360 - hawkeye_measured_left, hawkeye_measured_left),
    hawkeye_measured_right_standardized = if_else(pitch_hand == "L", 360 - hawkeye_measured_right, hawkeye_measured_right),
    movement_inferred_left_standardized = if_else(pitch_hand == "L", 360 - movement_inferred_left, movement_inferred_left),
    movement_inferred_right_standardized = if_else(pitch_hand == "L", 360 - movement_inferred_right, movement_inferred_right)
  ) |>
  select(
    -hawkeye_measured_left, -hawkeye_measured_right,
    -movement_inferred_left, -movement_inferred_right,
    -pitch_hand
  )

print("Created spin_joined1")
head(spin_joined1)

########## Calculate actual movement components
# - Use trigonometric functions to calculate the x and z components of the movement.
# - Convert angles from degrees to radians as trigonometric functions in R use radians.
# - Negative sign in z-component accounts for the coordinate system used.

# Calculate actual movement
# Remember that it's standardized to RHP
# (i.e. armside movement is (-) x_movement)
spin_joined2 <- spin_joined1 %>%
  mutate(
    movement_inches_left_z = -cos(movement_inferred_left_standardized * (pi / 180)), # cos() uses radians
    movement_inches_left_x = sin(movement_inferred_left_standardized * (pi / 180)),  # sin() uses radians
    movement_inches_right_z = -cos(movement_inferred_right_standardized * (pi / 180)),
    movement_inches_right_x = sin(movement_inferred_right_standardized * (pi / 180))
  )

print("Created spin_joined2")
head(spin_joined2)

########## Calculate expected movement components
# - Adjust the actual movement components by the ratio of 'active_spin' to 'alan_active_spin_pct'.
# - This gives the expected movement based on the spin efficiency.
# - Note: There might be a flaw if 'active_spin' is based on average 'gyro_deg' rather than typical values.

# Calculate expected movement
# POSSIBLE FLAW: active_spin might be based on average_gyro_deg; not *average* active spin, but *typical*
spin_joined3 <- spin_joined2 %>%
  mutate(
    exp_movement_inches_left_z = movement_inches_left_z * (active_spin_left / alan_active_spin_pct_left),
    exp_movement_inches_left_x = movement_inches_left_x * (active_spin_left / alan_active_spin_pct_left),
    exp_movement_inches_right_z = movement_inches_right_z * (active_spin_right / alan_active_spin_pct_right),
    exp_movement_inches_right_x = movement_inches_right_x * (active_spin_right / alan_active_spin_pct_right)
  )

print("Created spin_joined3")
head(spin_joined3)

########## Calculate gyro angle (gyro_deg)
# - Calculate 'gyro_deg' by taking the arccosine of 'active_spin' and converting from radians to degrees.
# - For reverse-gyro pitches, set 'gyro_deg' to negative.
# - Note: There might be a flaw if 'active_spin' is based on average 'gyro_deg' rather than typical values.

# Calculate gyro angle
# Reverse-gyro is (-)
# POSSIBLE FLAW: active_spin might be based on average_gyro_deg; not *average* active spin, but *typical*
spin_joined4 <- spin_joined3 %>%
  mutate(
    gyro_deg_left = acos(active_spin_left) * 180 / pi,
    gyro_deg_right = acos(active_spin_right) * 180 / pi
  ) |>
  mutate(
    gyro_deg_left = if_else(reverse_gyro_left == 1, -gyro_deg_left, gyro_deg_left),
    gyro_deg_right = if_else(reverse_gyro_right == 1, -gyro_deg_right, gyro_deg_right)
  )

print("Created spin_joined4")
head(spin_joined4)

########## Calculate predictor and response variables for the linear model
# - Predictors include differences in release speed, spin direction, spin rate, arm angle, and 'ssw_tilt'.
# - 'ssw_tilt' is the difference between the inferred spin axis and the measured spin axis.
# - Response variable is the difference in 'gyro_deg' between the two pitches.

# Calculate predictors & response
# POSSIBLE FLAW: not considering reverse-gyro (spin_direction_diff)
# Predictors: release_speed_diff, spin_direction_diff, spin_rate_diff, arm_angle_diff, ssw_tilt
# (Can't use ssw_xy instead of tilt because we don't have that pitch-by-pitch)
# Response: gyro_deg_diff
# (Going off of the right; predicting gyro_deg_diff for right)
spin_joined5 <- spin_joined4 %>%
  mutate(
    release_speed_diff = release_speed_right - release_speed_left,
    spin_direction_diff = hawkeye_measured_right_standardized - hawkeye_measured_left_standardized,
    spin_rate_diff = spin_rate_right - spin_rate_left,
    arm_angle_diff = avg_estimated_arm_angle_right - avg_estimated_arm_angle_left,
    ssw_tilt = movement_inferred_right_standardized - hawkeye_measured_right_standardized, # (+) is clockwise (ignored by abs())
    gyro_deg_diff = gyro_deg_right - gyro_deg_left
  )

print("Created spin_joined5")
head(spin_joined5)



##################################################################################################
# Step 3: Linear Model
# - Calculate weights based on the number of pitches.
# - Separate data into different pitch groups ('Fastball', 'Breaking', 'Offspeed') and reverse-gyro pitches.
# - Reverse-gyro pitches are separated because their behavior is opposite.

########## Linear model
# Calculate the weights
# Separate reverse-gyro pitches because they're opposite
Fastballs <- spin_joined5 %>%
  filter(pitch_group == "Fastball") %>%
  mutate(weight = n_pitches_left + n_pitches_right)

Breakings <- spin_joined5 %>%
  filter(pitch_group == "Breaking") %>%
  mutate(weight = n_pitches_left + n_pitches_right)

GOffspeeds <- spin_joined5 %>%
  filter(pitch_group == "Offspeed",
         reverse_gyro_right == 0) %>%
  mutate(weight = n_pitches_left + n_pitches_right) 

ROffspeeds <- spin_joined5 %>% # Reverse-gyro
  filter(pitch_group == "Offspeed",
         reverse_gyro_right == 1) %>%
  mutate(weight = n_pitches_left + n_pitches_right)

########## Fit weighted linear models for each pitch group
# - Use the calculated weights in the models.
# - The models predict 'gyro_deg_diff' based on the predictor variables.

# Fit weighted linear models
lin_model_Fastball <- lm(gyro_deg_diff ~ release_speed_diff + spin_direction_diff + spin_rate_diff + 
                           arm_angle_diff + ssw_tilt, 
                         data = Fastballs, weights = weight)

lin_model_Breaking <- lm(gyro_deg_diff ~ release_speed_diff + spin_direction_diff + spin_rate_diff + 
                           arm_angle_diff + ssw_tilt,  
                         data = Breakings, weights = weight)

lin_model_GOffspeed <- lm(gyro_deg_diff ~ release_speed_diff + spin_direction_diff + spin_rate_diff + 
                            arm_angle_diff + ssw_tilt, 
                          data = GOffspeeds, weights = weight)

lin_model_ROffspeed <- lm(gyro_deg_diff ~ release_speed_diff + spin_direction_diff + spin_rate_diff + 
                            arm_angle_diff + ssw_tilt, 
                          data = ROffspeeds, weights = weight)




##################################################################################################
# Step 4: Analysis


# Do more STA302 analyses



########## Display model summaries
summary(lin_model_Fastball) # R^2 = 0.3801
summary(lin_model_Breaking) # R^2 = 0.6783
summary(lin_model_GOffspeed) # R^2 = 0.0656 NO CORRELATION
summary(lin_model_ROffspeed) # R^2 = 0.0665 NO CORRELATION


########## Anova test
# Remove each variable
lin_model_reduced <- lm(gyro_deg_diff ~ release_speed_diff + spin_direction_diff + arm_angle_diff, data=Fastballs)
anova(lin_model_reduced, lin_model_Fastball)


########## Plotting

# # Add predicted values
# Fastballs <- Fastballs %>%
#   mutate(predicted = predict(lin_model_Fastball, newdata = Fastballs))
# 
# ggplot(Fastballs, aes(x = gyro_deg_diff, y = predicted)) +
#   geom_point(alpha = 0.5, color = "blue") +
#   geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") + # Reference line (y = x)
#   labs(
#     x = "Actual gyro_deg_diff",
#     y = "Predicted gyro_deg_diff",
#     title = "Predicted vs. Actual gyro_deg_diff for Fastballs"
#   ) +
#   theme_minimal()



##################################################################################################
# Step 5 Calculate predictor variables for 'savant_data' dataset
# - Prepare 'savant_used0' by renaming the 'avg_estimated_arm_angle' column.
# - Join 'savant_data' with 'savant_used0' to add 'pitch_type_average_arm_angle'.

# Calculate Predictor variables
# Predictors: release_speed_diff, spin_direction_diff, spin_rate_diff, arm_angle_diff, ssw_tilt

########## Get arm_angle data
savant_used0 <- savant_used |>
  rename(pitch_type_average_arm_angle = avg_estimated_arm_angle)

savant_data <- savant_data %>%
  left_join(savant_used0, by = c("player_name","year","pitch_type"))

# Add reverse-gyro information to 'savant_data'
# - From 'spin_joined0', extract reverse-gyro information and prepare 'reverse_gyro0'.
# - Join 'reverse_gyro0' with 'savant_data' to add 'reverse_gyro' indicator.
# - For pitch types not present in 'api_pitch_type', manually assign 'reverse_gyro' values based on knowledge of pitch types.
# - Note: Some sinkers have reverse-gyro SSW, which will be accounted for later.

########## Add (bool) reverse_gyro 
reverse_gyro0 <- spin_joined0 |>
  group_by(player_name, pitch_type_right, year_right) |>
  select(player_name, pitch_type_right, year_right, reverse_gyro_right) |>
  rename(pitch_type = pitch_type_right, 
         year = year_right, 
         reverse_gyro = reverse_gyro_right) |>
  distinct()

print(filter(reverse_gyro0), n = 100) # reverse_gyro0: 12681 rows (removed "Other" pitch group from spin)

savant_data <- savant_data %>%
  left_join(reverse_gyro0, by = c("year","player_name","pitch_type"))

# pitch_type not categorized in api_pitch_type:
#   Fastballs (FA) have gyro
#   Slow Curve (CS) have gyro
#   Knuckle Curve (KC) have gyro
#   Screwballs (SC) have reverse-gyro
#   Knuckleballs (KN) put as gyro
#   Forkballs (FO) have gyro
#   Eephus (EP) have gyro
savant_data <- savant_data |>
  mutate(reverse_gyro = if_else(pitch_type %in% c("FA","KC","CS","KN","FO","EP","PO"), 0,
                                if_else(pitch_type %in% c("SC"), 1, reverse_gyro)))
# Some sinkers (i.e. Josh Hader) have reverse-gyro SSW—accounted for below

# Merge pitch type averages with 'savant_data'
# - From 'spin1', calculate 'gyro_deg' and select relevant columns.
# - Rename columns to indicate they are pitch type averages.
# - Join these averages with 'savant_data'.

# Get average release_speed, spin_direction, spin_rate
spin0 <- spin1 |>
  mutate(gyro_deg = acos(active_spin) * 180 / pi) |> 
  select("year","player_name","pitch_type","pitch_group","release_speed","spin_rate","hawkeye_measured","gyro_deg") |>
  rename(pitch_type_release_speed = release_speed,
         pitch_type_spin_rate = spin_rate,
         pitch_type_spin_axis = hawkeye_measured,
         pitch_type_gyro_deg = gyro_deg) # If reverse-gyro, (-); addressed below

savant_data <- savant_data %>%
  left_join(spin0, by = c("player_name","year","pitch_type"))

########## Calculate predictor variables in 'savant_data'
# - Standardize 'spin_axis' to RHP by adjusting for left-handed pitchers.
# - Calculate 'inferred_spin_axis_standardized' based on 'pfx_x' and 'pfx_z' (pitch movement components).
# - Compute differences between actual values and pitch type averages for release speed, spin direction, spin rate, and arm angle.
# - Calculate 'ssw_tilt' as the difference between inferred spin axis and measured spin axis.
savant_data <- savant_data %>%
  mutate(
    # Standardize to RHP
    spin_axis_standardized = if_else(p_throws == "L", 360 - spin_axis, spin_axis),
    inferred_spin_axis_standardized = if_else(p_throws == "L", atan2(pfx_x, pfx_z) * (180 / pi) + 180,
                                              atan2(-pfx_x, pfx_z) * (180 / pi) + 180)
  ) %>%
  mutate(
    release_speed_diff = release_speed - pitch_type_release_speed,
    spin_direction_diff = pitch_type_spin_axis - spin_axis_standardized,
    spin_rate_diff =  release_spin_rate - pitch_type_spin_rate,
    arm_angle_diff = estimated_arm_angle - pitch_type_average_arm_angle,
    ssw_tilt = inferred_spin_axis_standardized - spin_axis_standardized # (+) is clockwise
  ) 

head(savant_data)

########## Adjust 'reverse_gyro' for specific cases (e.g., Josh Hader's sinker)
# - Identify sinkers ('SI') with counter-clockwise SSW (positive 'ssw_tilt') as reverse-gyro pitches.
# - Specifically adjust for Josh Hader's sinker.
# - Update 'reverse_gyro' accordingly.

# Additional: If SI & counter-clockwise SSW, reverse-gyro
# List: Josh Hader
SI <- savant_data %>%
  filter(pitch_type == "SI", player_name == "Hader, Josh") %>%
  group_by(player_name, year) %>%
  summarise(ssw_tilt = mean(ssw_tilt), .groups = "drop") %>%
  mutate(reverse_gyro_si = if_else(ssw_tilt > 0, 1, 0))

savant_data <- savant_data %>%
  left_join(SI %>% select(player_name, year, reverse_gyro_si), 
            by = c("player_name", "year")) %>%
  mutate(reverse_gyro = if_else(pitch_type == "SI" & !is.na(reverse_gyro_si), reverse_gyro_si, reverse_gyro)) %>%
  select(-reverse_gyro_si)  # Remove the temporary column

# Adjust 'pitch_type_gyro_deg' for reverse-gyro pitches
# - For reverse-gyro pitches, set 'pitch_type_gyro_deg' to negative.

# Additional: If reverse-gyro, gyro_deg is (-)
savant_data <- savant_data %>%
  mutate(pitch_type_gyro_deg = if_else(reverse_gyro == 1, -pitch_type_gyro_deg, pitch_type_gyro_deg))



##################################################################################################
# Step 6: Apply the linear models to 'savant_data'
# - Update 'pitch_group' based on 'pitch_type'.
# - Note: Offspeed pitches are excluded from modeling due to low correlation.

# Apply the model to savant_data
# Give up on Offspeeds since no correlation

########## Additional: Update pitch_groups
savant_data <- savant_data %>%
  mutate(pitch_group = if_else(pitch_type %in% c("FA","FC","FF","SI"), "Fastball",
                               if_else(pitch_type %in% c("CS","CU","KC","SL","ST","SV"), "Breaking",
                                       if_else(pitch_type %in% c("CH","FO","FS","SC"), "Offspeed", NA))))

########## Predict 'estimated_gyro_deg' using the linear models
# - For 'Fastball' and 'Breaking' pitch groups, add the predicted 'gyro_deg_diff' to 'pitch_type_gyro_deg'.
# - For 'Offspeed' pitches, use 'pitch_type_gyro_deg' directly (since models are not reliable).
# - For other pitches, set 'estimated_gyro_deg' to NA.

# Make predictions about gyro_deg
# For FO, SC, estimated/pitch_type_gyro_deg unavailable—need to find ways to scrape individually or something
savant_data <- savant_data %>%
  mutate(
    estimated_gyro_deg = case_when(
      pitch_group == "Fastball" ~ pitch_type_gyro_deg + predict(lin_model_Fastball, newdata = savant_data),
      pitch_group == "Breaking" ~ pitch_type_gyro_deg + predict(lin_model_Breaking, newdata = savant_data),
      pitch_group == "Offspeed" ~ pitch_type_gyro_deg,
      TRUE ~ NA_real_ # could leave this as 0, but didn't make too much sense
    )
  )

head(savant_data)



##################################################################################################
# Step 7: Analysis

########## Check distributions
# ggplot(savant_data %>% filter(pitch_group == "Fastball"), aes(x = pitch_type_gyro_deg)) +
#   geom_histogram(aes(fill = "Pitch Type Gyro Deg"), alpha = 0.5, bins = 30, color = "black") +
#   geom_histogram(aes(x = estimated_gyro_deg, fill = "Estimated Gyro Deg"), alpha = 0.5, bins = 30, color = "black") +
#   scale_fill_manual(name = "Gyro Deg Type", values = c("Pitch Type Gyro Deg" = "blue", "Estimated Gyro Deg" = "red")) +
#   labs(
#     x = "Gyro Deg",
#     y = "Frequency",
#     title = "Fastball: Estimated vs. Pitch Type Gyro Deg"
#   ) +
#   theme_minimal()



##################################################################################################
# Step 8: Save the updated 'savant_data' to a CSV file

########## Write CSV
write.csv(savant_data, "Dataset/savant_data_2020-2024_updated.csv")
