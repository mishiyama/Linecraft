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
import matplotlib.pyplot as plt 



# Sample data points (x: independent, y: dependent)
x = np.array([500, 1000, 1500, 2000])
y = np.array([50, 10, 150, 200])

# Function to calculate slope (m) and intercept (b)
def best_fit_line(x, y):
    m = (np.mean(x*y) - np.mean(x) * np.mean(y)) / (np.mean(x**2) - np.mean(x)**2)
    b = np.mean(y) - m * np.mean(x)
    return m, b

# Get slope and intercept
m, b = best_fit_line(x, y)

# Print the equation
print(f"Best Fit Line: y = {m:.2f}x + {b:.2f}")

# Predict y values
y_pred = m * x + b

# Plot original data and best fit line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='Best Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression - Best Fit Line')
plt.legend()
plt.grid(True)
plt.show()


# Predict y for a specific x value
def predict_values(x_value,m,b):
 return m*x_value+b

x_value=float(input("Enter the value for x: "))
y_predicted_value=predict_values(x_value,m,b)
print(f"For x={x_value} , y={y_predicted_value}")


