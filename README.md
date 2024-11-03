## StuffModelML

### Raw_data
- savant_data_YYYY: has all the pitch_by_pitch data
- arm_angles_YYYY: has all the arm angle data (per year, per pitcher)
- spin_directions_YYYY: has all the spin direction/efficiency data (per year, per pitch type, per pitcher) 
    - (savant_data has direction data, but not efficiency)


## Checklist
- Filter for pitch types for which pitchers have thrown 10+ of each year (due to arm_angles and spin_directions dataset restriction)
