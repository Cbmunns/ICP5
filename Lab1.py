import pandas as pd
import matplotlib.pyplot as plt

# Read in the csv file
df = pd.read_csv("garage.csv")

# Plot just GarageArea by SalePrice to visually find outliers 
df.plot(kind='scatter', x='GarageArea', y='SalePrice')
plt.title('Raw Data')
plt.show()

# Set the width(Garage area) to the most common sizes
first_drop = df[(df['GarageArea'] < 1100) & (df['GarageArea'] > 140)]
# Set height(Sale price) to remove excessivly high prices
no_outliers = first_drop[(first_drop['SalePrice'] < 500000)]

# Plot the updated and focused data
no_outliers.plot(kind='scatter', x='GarageArea', y='SalePrice')
plt.title('Data with no outliers')
plt.show()

# Sources

# https://stackoverflow.com/questions/14300137/making-matplotlib-scatter-plots-from-dataframes-in-pythons-pandas
# https://www.earthdatascience.org/courses/intro-to-earth-data-science/scientific-data-structures-python/pandas-dataframes/indexing-filtering-data-pandas-dataframes/
# preprocessing.py source code provided


