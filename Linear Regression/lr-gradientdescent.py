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

import numpy as np 

# Data
X = np.array([
    [2104, 3],
    [1600, 3],
    [2400, 3],
    [1416, 2]
])
y = np.array([399900, 329900, 369000, 232000])

m=len(y) #Number of data sets 

# Hyperparameters
alpha = 0.00000001
iterations = 1000

#Add intercept term 
X_b=np.c_[np.ones((m,1)),X]



def gradient_descent():
 # Initialize parameters
 theta = np.zeros(X_b.shape[1])
 for i in range(iterations):
  y_pred=X_b.dot(theta) #predict values
  error=y_pred-y
  gradients=(1/m)*X_b.T.dot(error)
  theta=theta - alpha * gradients

  if i % 100 == 0:
   cost = (1/(2*m)) * np.sum(error**2)
   print(f"Iteration {i}: Cost {cost:.2f}")
 return theta

theta=gradient_descent()

#To predict values for new input 
house_area=int(input("Enter the area of house in sq ft : "))
bedroom_no=int(input("Enter number of bedroom house contains : "))
newhouse_data=np.array([1,house_area,bedroom_no])

predicted_price=newhouse_data.dot(theta)
print(f"Predicted price: ${predicted_price:.2f}")








