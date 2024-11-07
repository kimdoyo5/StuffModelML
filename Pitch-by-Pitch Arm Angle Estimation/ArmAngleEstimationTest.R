library(ggplot2)
library(dplyr)

data <- read.csv("Dataset/savant_data_2020-2024_updated.csv",header=T)

data1 <- data |>
  filter(player_name == "Buehler, Walker")

ggplot(data1) +
  geom_histogram(aes(x = estimated_arm_angle), binwidth = 1, fill = "blue", color = "black", alpha = 0.5) +
  geom_histogram(aes(x = precise_average_arm_angle), binwidth = 1, fill = "red", color = "black", alpha = 0.5) +
  labs(title = "Overlapping Histograms of Arm Angle and Precise Arm Angle",
       x = "Angle", y = "Frequency") +
  theme_minimal()


data2 <- data1 |>
  select(game_date, year, precise_average_arm_angle, estimated_arm_angle) |>
  group_by(year) |>
  summarise(precise_average_arm_angle = mean(precise_average_arm_angle, na.rm=T),
            estimated_average_arm_angle = mean(estimated_arm_angle, na.rm=T))
data2
