import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read file in
df = pd.read_csv('city.csv')

# Handle missing data
data = df.select_dtypes(include=[np.number]).interpolate().dropna()

# # test data frame for scrubbing
# test = data.drop(['Id'], axis=1)
# # Go through each column and figure out which data point in each had the greatest correlation
# for column in test:
#     print(test[[column,'revenue']].groupby([column], as_index=False).mean().sort_values(by=column, ascending=False))

# Set the x axis to the 5 most correlated values
y = np.log(df['revenue'])
X = data[['P1', 'P18', 'P25', 'P29', 'P34']]
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size =.33)

# Set up the Regression model
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Display the R2 value
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
# Display the RMSE value
print ('RMSE is: \n', mean_squared_error(y_test, predictions))



# Sources
# regression.py
# Lab2.py
# ICP4