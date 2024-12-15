import pandas as pd
import numpy as np

# Number of rows in the dataset
num_rows = 1000

# Generate random sample data for the dataset
locations = ['Zone A', 'Zone B', 'Zone C', 'Zone D']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
waste_types = ['Organic', 'Plastic', 'Paper', 'Metal']
collection_frequency = ['Daily', 'Weekly', 'Bi-weekly']
weather_conditions = ['Sunny', 'Rainy', 'Cloudy']
population_density_range = (1000, 10000)  # People per square kilometer
cost_range = (50, 500)  # Cost of waste collection in local currency
vehicle_types = ['Large', 'Medium', 'Small']
recycling_percentage_range = (0, 100)  # Percentage of waste recycled

# Randomly generate data for each column
data = {
    'Location': np.random.choice(locations, num_rows),
    'Day of Week': np.random.choice(days_of_week, num_rows),
    'Temperature (°C)': np.random.uniform(15, 35, num_rows),
    'Humidity (%)': np.random.uniform(30, 80, num_rows),
    'Waste Type': np.random.choice(waste_types, num_rows),
    'Waste Amount (kg)': np.random.uniform(1, 100, num_rows),
    'Population Density (people/km²)': np.random.uniform(population_density_range[0], population_density_range[1], num_rows),
    'Collection Frequency': np.random.choice(collection_frequency, num_rows),
    'Day of Month': np.random.randint(1, 31, num_rows),
    'Special Event Flag': np.random.choice([0, 1], num_rows),  # 0: No Event, 1: Event
    'Waste Collection Cost': np.random.uniform(cost_range[0], cost_range[1], num_rows),
    'Recycling Percentage': np.random.uniform(recycling_percentage_range[0], recycling_percentage_range[1], num_rows),
    'Collection Vehicle Type': np.random.choice(vehicle_types, num_rows),
    'Weather Condition': np.random.choice(weather_conditions, num_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Balance the dataset manually: Ensure equal representation of each 'Waste Type'
df_majority = df[df['Waste Type'] == 'Organic']
df_minority = df[df['Waste Type'] != 'Organic']

# Upsample the minority classes (Plastic, Paper, Metal) to match the majority class
df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)

# Combine the majority class with the upsampled minority classes
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle the dataset to mix the classes
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a CSV file
df_balanced.to_csv('ecoFlow_waste_management_balanced.csv', index=False)

print(df_balanced.head())  # Display the first few rows of the balanced dataset
