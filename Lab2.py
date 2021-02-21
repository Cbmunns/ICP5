import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read file in
df = pd.read_csv('city.csv')
#print(df.head())

# Convert city group to num for it to work in the model
data = df.replace({"Big Cities": 1, 'Other': 2})
#print(data.head())

# Convert type of restraunt to num for it to work in the model
data = data.replace({'FC': 1, 'IL': 2, 'DT': 3, 'MB': 4})
#print(data.head())

# Handle missing data
data = data.select_dtypes(include=[np.number]).interpolate().dropna()
print(data.head())

# Set up the data for splitting
y = np.log(df['revenue'])
X = data.drop(['Id', 'revenue'], axis=1)

# Split up the data accordingly
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size =.33)

# Set up the regression model
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Display the R2 score
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)

# Display the RMSE score
print ('RMSE is: \n', mean_squared_error(y_test, predictions))







# Sources
# regression.py source file
# https://heartbeat.fritz.ai/implementing-multiple-linear-regression-using-sklearn-43b3d3f2fe8b