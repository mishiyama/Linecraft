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

# #########IMPLEMENTAION OF MULTIPLE LINEAR REGRESSION##########


# Importing required dependencies

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#To read the data from CSV file
df=pd.read_csv('realestate_data.csv')
df.drop(columns=['House ID'],inplace=True)

#Assign independent and dependent features
X = df.drop(columns=['Price (USD)'])  # All columns except target
y=df['Price (USD)']

#Test-Train split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


#Standardization 
scaler=StandardScaler()                    # z-score normalization (useful when features in data set have different units or scales) 
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)           # using transform in test data to prevent data leakage (no info regarding train data to be revealed to test)

#Applying Simple linear regression 
regression=LinearRegression()
regression.fit(X_train,y_train)
print("Coefficent: ",regression.coef_)
print("Intercept: ",regression.intercept_)


y_pred_train = regression.predict(X_train)
y_pred_test = regression.predict(X_test)

print("Train R² Score:", r2_score(y_train, y_pred_train))
print("Test R² Score:", r2_score(y_test, y_pred_test))
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("MSE:", mean_squared_error(y_test, y_pred_test))
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.show()


# Function to predict price for new data
def predict_new_price():
    print("\n### Enter New House Details ###")
    try:
        bedrooms = int(input("Enter number of bedrooms: "))
        sqft = float(input("Enter square footage: "))
        age = float(input("Enter age of house (years): "))
        distance = float(input("Enter distance to city center (km): "))

        # Create a DataFrame for the new input
        new_data = pd.DataFrame([[bedrooms, sqft, age, distance]],
                                columns=['Bedrooms', 'Square Footage', 'Age of House (years)', 'Distance to City Center (km)'])

        # Standardize the new input using previously fitted scaler
        new_data_scaled = scaler.transform(new_data)

        # Predict using trained regression model
        predicted_price = regression.predict(new_data_scaled)

        print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")
    except Exception as e:
        print("Error in input:", e)

# Call the function to test
predict_new_price()


