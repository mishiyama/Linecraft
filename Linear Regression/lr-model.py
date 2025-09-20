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

# Importing required dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#To read the data 
df=pd.read_csv('weight_height_data.csv')

#Assign independent and dependent feature 
X=df[['Weight (kg)']]  #independent feature must be 2D 
y=df['Height (cm)']    #dependent feature must be 1D

#Train-Test split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


#Standardization 
scaler=StandardScaler()                    # z-score normalization (useful when features in data set have different units or scales) 
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)            # using transform in test data to prevent data leakage (no info regarding train data to be revealed to test)

#Applying Simple linear regression 
regression=LinearRegression()
regression.fit(X_train,y_train)
print("Coefficent: ",regression.coef_)
print("Intercept: ",regression.intercept_)

#Plot best fit line (training data)
plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train))  #predict(X_train) refers to y predict 


#Prediction for test data
y_pred=regression.predict(X_test)

#Performance Metrics 
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)

#R-sqaured 
score=r2_score(y_test,y_pred)
print(score)

#Adjusted R-sqaured
r_score=1-(1-score)*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)
print(r_score)

#Applying OLS LinearRegression
model=sm.OLS(y_train,X_train).fit()
prediction=model.predict(X_test)
print(prediction)

#Summary
print(model.summary())

#Prediction of new entered new_value
def predict_value(new_weight_value):
 new_weight_df = pd.DataFrame([[new_weight_value]], columns=['Weight (kg)'])
 new_height_value=regression.predict(scaler.transform(new_weight_df))
 return new_height_value[0]


new_weight_value=float(input("Enter the weight (kg): "))
new_height_value = predict_value(new_weight_value)
print(f"For weight {new_weight_value} the height would be {new_height_value}cm ")




