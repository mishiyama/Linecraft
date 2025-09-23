# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# #########IMPLEMENTATION OF POLYNOMIAL REGRESSION##########


# Importing dependices 

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


#Quadratic equation used   y=0.5x^2 + 1.5x + 2 + outliers
X = 6 * np.random.rand(100,1) - 3
y = 0.5 * X**2 + 1.5*X + 2 +np.random.randn(100,1)
plt.scatter(X,y,color='g')
plt.xlabel("X DATASET")
plt.ylabel("Y DATASET")


#Train Test split 
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Implementation of Simple Linear Regression 
regression_1=LinearRegression()
regression_1.fit(X_train,y_train)
score=r2_score(y_test,regression_1.predict(X_test))
print(score)

#Visualize the model 
plt.plot(X_train , regression_1.predict(X_train),color='r')
plt.scatter(X_train,y_train)
plt.xlabel("X DATASET ")
plt.ylabel("Y ")

#Implementation of polynomial transformation
poly=PolynomialFeatures(degree=2,include_bias=True)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

#Performance metrics 
regression=LinearRegression()
regression.fit(X_train_poly,y_train)
y_pred=regression.predict(X_test_poly)
score=r2_score(y_test,y_pred)
print(score)


print(regression.coef_)
print(regression.intercept_)
plt.scatter(X_train,regression.predict(X_train_poly))
plt.scatter(X_train,y_train)


# User Input for Prediction
try:
    user_input = float(input("Enter a new X value to predict Y using Polynomial Regression: "))
    user_input_array = np.array([[user_input]])
    user_input_poly = poly.transform(user_input_array)
    predicted_y = regression.predict(user_input_poly)
    print(f"Predicted Y for X = {user_input} is: {predicted_y[0][0]:.3f}")
except ValueError:
    print("Invalid input. Please enter a numerical value.")
