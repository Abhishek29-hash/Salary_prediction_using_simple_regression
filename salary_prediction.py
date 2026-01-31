import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# importing the dataset
df = pd.read_csv("Data.csv")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state= 0)

# training the simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# predicting the test set results 
y_pred = regressor.predict(x_test)
# print(y_pred)


# visualising the training set results
# plt.scatter(x_train, y_train, color='red') 
# plt.plot(x_train, regressor.predict(x_train), color='blue')
# plt.title("Salary vs Experience (Training set)")
# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
# plt.show()


# visualising the test set results
# plt.scatter(x_test, y_test, color='red') 
# plt.plot(x_train, regressor.predict(x_train), color='blue')
# plt.title("Salary vs Experience (Test set)")
# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
# plt.show()



# predicting a single value
print("Prediction for 6.5 years of experience is ", regressor.predict([[6.5]]))


# getting the model parameters
# like coefficient and intercept
print("regressor.coef_:", regressor.coef_)
print("regressor.intercept_:", regressor.intercept_)
