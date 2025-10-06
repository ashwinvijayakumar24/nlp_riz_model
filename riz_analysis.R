# Data
year <- c(
  2004,2004,2004, 2005,2005,2005, 2006,2006,2006, 2007,2007,2007,
  2008,2008,2008, 2009,2009,2009, 2010,2010,2010, 2011,2011,2011,
  2012,2012,2012, 2013,2013,2013, 2014,2014,2014, 2015,2015,2015,
  2016,2016,2016, 2017,2017,2017, 2018,2018,2018, 2019,2019,2019,
  2020,2020,2020, 2021,2021,2021, 2022,2022,2022, 2023,2023,2023
)

riz_score <- c(
  1,0,0, 2,1,1, 1,0,1, 0,0,0,
  1,2,1, 1,2,0, 1,0,0, 1,0,1,
  2,0,1, 0,1,1, 0,1,1, 1,2,0,
  0,2,0, 2,1,2, 2,3,1, 1,1,0,
  2,2,2, 1,1,2, 1,2,1, 1,2,2
)

# Normalize year
years_after_2004 <- year - 2004
df <- data.frame(Year = years_after_2004, Riz = riz_score)

# Add a flag for post-2014
df$Post2014 <- year >= 2015

# Fit linear model
model <- lm(Riz ~ Year, data = df)
slope <- round(coef(model)[2], 5)
intercept <- round(coef(model)[1], 5)
equation <- paste0("ŷ = ", intercept, " + ", slope, "x")

# Plot setup
par(mar = c(5, 5, 4, 2))
plot(df$Year, df$Riz,
     type = "n",
     main = "Muslim Misrepresentation in Bollywood from 2004–2023",
     xlab = "Movie Release Date (Years after 2004)",
     ylab = "Riz Score",
     xaxt = "n",  
     yaxt = "n",
     cex.main = 1,
     cex.lab = 1)

axis(side = 1, at = 0:19, labels = 0:19)
axis(side = 2, at = 0:3, labels = 0:3)



# Plot pre-2015 movies in black
points(df$Year[!df$Post2014], df$Riz[!df$Post2014],
       pch = 16, col = "black")

# Plot post-2014 movies in blue
points(df$Year[df$Post2014], df$Riz[df$Post2014],
       pch = 17, col = "blue")  # triangle shape for distinction

# Add red regression line
abline(model, col = "red", lwd = 1.5)

# Add regression equation
text(x = 13, y = 1.3, labels = equation, cex = 0.8, pos=3)

# Add legend
legend(
       x = -0.3,
       y = 3,
       cex = 0.8,
       legend = c("Pre-Modi", "Current"), 
       pch = c(16, 17), 
       col = c("black", "blue"))