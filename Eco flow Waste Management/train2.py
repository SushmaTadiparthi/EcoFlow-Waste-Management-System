import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model for waste generation forecasting
with open('waste_generation_forecasting_model.pkl', 'rb') as file:
    waste_gen_model = pickle.load(file)

# Sample data for insights and route optimization (you can replace with real data)
sample_waste_data = pd.DataFrame({
    'Location': ['Zone A', 'Zone B', 'Zone C', 'Zone D'],
    'Average Distance': [5, 10, 15, 20],  # In kilometers, for route optimization example
    'Waste Generated': [100, 200, 150, 180]  # In kg, sample past data for insights
})

# Function to handle Waste Generation Forecasting
def waste_generation_forecasting():
    # Clear previous widgets
    for widget in waste_forecasting_frame.winfo_children():
        widget.grid_forget()

    tk.Label(waste_forecasting_frame, text="Location (e.g., Zone A)").grid(row=0, column=0)
    location_entry = tk.Entry(waste_forecasting_frame)
    location_entry.grid(row=0, column=1)

    tk.Label(waste_forecasting_frame, text="Temperature (°C)").grid(row=1, column=0)
    temp_entry = tk.Entry(waste_forecasting_frame)
    temp_entry.grid(row=1, column=1)

    tk.Label(waste_forecasting_frame, text="Humidity (%)").grid(row=2, column=0)
    humidity_entry = tk.Entry(waste_forecasting_frame)
    humidity_entry.grid(row=2, column=1)

    tk.Label(waste_forecasting_frame, text="Day of Week (e.g., Monday)").grid(row=3, column=0)
    day_of_week_entry = tk.Entry(waste_forecasting_frame)
    day_of_week_entry.grid(row=3, column=1)

    def predict_waste():
        try:
            location = location_entry.get()
            temperature = float(temp_entry.get())
            humidity = float(humidity_entry.get())
            day_of_week = day_of_week_entry.get()

            day_of_week_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            location_map = {'Zone A': 0, 'Zone B': 1, 'Zone C': 2, 'Zone D': 3}

            day_of_week_val = day_of_week_map.get(day_of_week, -1)
            location_val = location_map.get(location, -1)

            if day_of_week_val == -1 or location_val == -1:
                raise ValueError("Invalid location or day of week. Please use valid inputs.")

            input_data = np.array([[temperature, humidity, day_of_week_val, location_val]])
            input_df = pd.DataFrame(input_data, columns=["Temperature", "Humidity", "DayOfWeek", "Location"])
            predicted_waste = waste_gen_model.predict(input_df)

            result_label.config(text=f"Predicted Waste Generation: {predicted_waste[0]:.2f} kg")
        
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    predict_button = tk.Button(waste_forecasting_frame, text="Predict Waste Generation", command=predict_waste)
    predict_button.grid(row=4, columnspan=2)

# Function for Route Optimization
def route_optimization():
    for widget in waste_forecasting_frame.winfo_children():
        widget.grid_forget()

    tk.Label(waste_forecasting_frame, text="Select Location for Route Optimization").grid(row=0, column=0)
    location_entry = tk.Entry(waste_forecasting_frame)
    location_entry.grid(row=0, column=1)

    def optimize_route():
        location = location_entry.get()
        if location in sample_waste_data['Location'].values:
            distance = sample_waste_data[sample_waste_data['Location'] == location]['Average Distance'].values[0]
            result_label.config(text=f"Optimized Route: {location} with distance {distance} km")
        else:
            messagebox.showerror("Invalid Location", "Please enter a valid location (e.g., Zone A, Zone B)")

    optimize_button = tk.Button(waste_forecasting_frame, text="Optimize Route", command=optimize_route)
    optimize_button.grid(row=1, columnspan=2)

# Function for Insights Generation
def generate_insights():
    for widget in waste_forecasting_frame.winfo_children():
        widget.grid_forget()

    # Calculate basic insights from the sample data
    avg_waste = sample_waste_data['Waste Generated'].mean()
    max_waste = sample_waste_data['Waste Generated'].max()
    min_waste = sample_waste_data['Waste Generated'].min()

    insights_text = (
        f"Waste Generation Insights:\n"
        f"Average Waste: {avg_waste:.2f} kg\n"
        f"Maximum Waste: {max_waste:.2f} kg\n"
        f"Minimum Waste: {min_waste:.2f} kg"
    )

    result_label.config(text=insights_text)

# Function to update the task selection based on user choice
def update_task_selection(*args):
    task = task_var.get()
    if task == 'Waste Generation Forecasting':
        waste_generation_forecasting()
    elif task == 'Route Optimization':
        route_optimization()
    elif task == 'Insights':
        generate_insights()
    else:
        result_label.config(text="Prediction Result will be shown here.")

# Main application setup
root = tk.Tk()
root.title("Waste Management Prediction")

# Dropdown menu to select task
task_var = tk.StringVar()
task_var.set("Select Task")
task_options = ["Select Task", "Waste Generation Forecasting", "Route Optimization", "Insights"]
task_menu = tk.OptionMenu(root, task_var, *task_options)
task_menu.grid(row=0, column=0, padx=10, pady=10)
task_var.trace("w", update_task_selection)

# Frame for task-specific inputs
waste_forecasting_frame = tk.Frame(root)
waste_forecasting_frame.grid(row=1, column=0, padx=10, pady=10)

# Label for displaying results
result_label = tk.Label(root, text="Prediction Result will be shown here.", font=("Helvetica", 14))
result_label.grid(row=2, column=0, padx=10, pady=10)

# Run the application
root.mainloop()
