import pandas as pd

savant_data = pd.read_csv("Dataset/savant_data_2020-2024_updated.csv")
print("savant_data loaded")

# Print all the column names
print(savant_data.columns.tolist())

# Remove unuseful columns
savant_data = savant_data.drop(
    columns=['batter','spin_dir','spin_rate_deprecated','break_angle_deprecated','break_length_deprecated',
             'pitcher_1','fielder_2','fielder_2_1','fielder_3','fielder_4','fielder_5','fielder_6','fielder_7','fielder_8','fielder_9',
             'effective_speed','home_score','away_score','post_home_score','post_away_score','inning','inning_topbot',
             'tfs_deprecated','tfs_zulu_deprecated','umpire','sv_id','bb_type','at_bat_number','babip_value','iso_value',
             'release_pos_y','type','game_type','game_year','if_fielding_alignment','of_fielding_alignment',
             'bat_score','fld_score','post_bat_score','post_fld_score','delta_home_win_exp','Unnamed..0','Unnamed: 0'],
    errors='ignore')
print("Update completed.")


# Save headers to a new CSV file
headers = savant_data.columns.tolist()
output_file = "Dataset/savant_data_2020-2024_updated_headers.csv"
pd.DataFrame(headers, columns=["Header"]).to_csv(output_file, index=False)
print(f"Headers have been saved to {output_file}")


# Save the updated savant data
savant_data.to_csv("Dataset/savant_data_2020-2024_updated.csv", index=False)

print("Exported data to csv.")