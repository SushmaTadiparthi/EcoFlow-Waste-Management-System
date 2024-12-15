import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the balanced dataset
df = pd.read_csv('ecoFlow_waste_management_balanced.csv')

# Encoding categorical features
label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Day of Week'] = label_encoder.fit_transform(df['Day of Week'])

# Waste Generation Forecasting Model
# Selecting features (input) and target (output)
X_waste = df[['Location', 'Temperature (°C)', 'Humidity (%)', 'Day of Week']]
y_waste = df['Waste Amount (kg)']

# Split the data into training and testing sets
X_train_waste, X_test_waste, y_train_waste, y_test_waste = train_test_split(X_waste, y_waste, test_size=0.2, random_state=42)

# Initialize and train the Waste Generation Forecasting model
waste_model = LinearRegression()
waste_model.fit(X_train_waste, y_train_waste)

# Predicting on the test set
y_pred_waste = waste_model.predict(X_test_waste)

# Evaluate the Waste Generation Forecasting model
mae_waste = mean_absolute_error(y_test_waste, y_pred_waste)
mse_waste = mean_squared_error(y_test_waste, y_pred_waste)
rmse_waste = np.sqrt(mse_waste)

print(f"Waste Generation Forecasting - MAE: {mae_waste}")
print(f"Waste Generation Forecasting - MSE: {mse_waste}")
print(f"Waste Generation Forecasting - RMSE: {rmse_waste}")

# Route Optimization Model
# Example synthetic features for Route Optimization
df['Truck Location (x)'] = np.random.uniform(0, 100, len(df))  # Example x-coordinates for truck
df['Truck Location (y)'] = np.random.uniform(0, 100, len(df))  # Example y-coordinates for truck
df['Traffic Delay (min)'] = np.random.uniform(0, 30, len(df))  # Traffic delay in minutes

# Selecting input features for Route Optimization
X_route = df[['Truck Location (x)', 'Truck Location (y)', 'Waste Amount (kg)', 'Traffic Delay (min)', 'Population Density (people/km²)']]
y_route = df['Waste Collection Cost']  # You can replace this with time or distance to collect

# Split the data into training and testing sets for Route Optimization
X_train_route, X_test_route, y_train_route, y_test_route = train_test_split(X_route, y_route, test_size=0.2, random_state=42)

# Initialize and train the Route Optimization model
route_model = LinearRegression()
route_model.fit(X_train_route, y_train_route)

# Predicting on the test set for Route Optimization
y_pred_route = route_model.predict(X_test_route)

# Evaluate the Route Optimization model
mae_route = mean_absolute_error(y_test_route, y_pred_route)
mse_route = mean_squared_error(y_test_route, y_pred_route)
rmse_route = np.sqrt(mse_route)

print(f"Route Optimization - MAE: {mae_route}")
print(f"Route Optimization - MSE: {mse_route}")
print(f"Route Optimization - RMSE: {rmse_route}")

# Insights Model (Decision Tree)
# Selecting input features for Insights
X_insights = df[['Location', 'Temperature (°C)', 'Humidity (%)', 'Day of Week']]
y_insights = df['Waste Amount (kg)']

# Split the data into training and testing sets for Insights model
X_train_insights, X_test_insights, y_train_insights, y_test_insights = train_test_split(X_insights, y_insights, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model for Insights
insights_model = DecisionTreeRegressor(random_state=42)
insights_model.fit(X_train_insights, y_train_insights)

# Predicting on the test set for Insights
y_pred_insights = insights_model.predict(X_test_insights)

# Evaluate the Insights model
mae_insights = mean_absolute_error(y_test_insights, y_pred_insights)
mse_insights = mean_squared_error(y_test_insights, y_pred_insights)
rmse_insights = np.sqrt(mse_insights)

print(f"Insights - MAE: {mae_insights}")
print(f"Insights - MSE: {mse_insights}")
print(f"Insights - RMSE: {rmse_insights}")

# Save the trained models as pickle files
with open('waste_generation_forecasting_model.pkl', 'wb') as f:
    pickle.dump(waste_model, f)

with open('route_optimization_model.pkl', 'wb') as f:
    pickle.dump(route_model, f)

with open('insights_model.pkl', 'wb') as f:
    pickle.dump(insights_model, f)

print("Models have been saved as pickle files.")
