from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Bin data
bins_data = {
    "Bin 1": {"filled": 30, "capacity": 100},
    "Bin 2": {"filled": 70, "capacity": 100},
    "Bin 3": {"filled": 45, "capacity": 100},
    "Bin 4": {"filled": 90, "capacity": 100},
    "Bin 5": {"filled": 10, "capacity": 100},
    "Bin 6": {"filled": 50, "capacity": 100},
}

# Load the pre-trained waste generation forecasting model
with open('waste_generation_forecasting_model.pkl', 'rb') as file:
    waste_gen_model = pickle.load(file)

# Sample data for insights and route optimization
sample_waste_data = pd.DataFrame({
    'Location': ['Zone A', 'Zone B', 'Zone C', 'Zone D'],
    'Average Distance': [5, 10, 15, 20],  # Example distances in km
    'Waste Generated': [100, 200, 150, 180]  # Example past data in kg
})


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict_waste', methods=['GET', 'POST'])
def predict_waste():
    if request.method == 'GET':
        return render_template('predict_waste.html')
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            location = data.get('location')
            temperature = float(data.get('temperature', 0))
            humidity = float(data.get('humidity', 0))
            day_of_week = data.get('day_of_week')

            if not location or not day_of_week:
                return jsonify({"error": "Location and day of week are required."}), 400

            day_of_week_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            location_map = {'Zone A': 0, 'Zone B': 1, 'Zone C': 2, 'Zone D': 3}

            day_of_week_val = day_of_week_map.get(day_of_week, -1)
            location_val = location_map.get(location, -1)

            if day_of_week_val == -1 or location_val == -1:
                return jsonify({"error": "Invalid location or day of week."}), 400

            input_data = np.array([[temperature, humidity, day_of_week_val, location_val]])
            input_df = pd.DataFrame(input_data, columns=["Temperature", "Humidity", "DayOfWeek", "Location"])
            predicted_waste = waste_gen_model.predict(input_df)

            return jsonify({"predicted_waste": f"{predicted_waste[0]:.2f} kg"})

        except Exception as e:
            return jsonify({"error": str(e)}), 400


@app.route('/optimize_route', methods=['GET', 'POST'])
def optimize_route():
    if request.method == 'GET':
        return render_template('optimize_route.html')
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            location = data.get('location')
            if not location:
                return jsonify({"error": "Location is required."}), 400

            if location in sample_waste_data['Location'].values:
                distance = sample_waste_data[sample_waste_data['Location'] == location]['Average Distance'].values[0]
                return jsonify({"optimized_route": f"{location} with distance {distance} km"})
            else:
                return jsonify({"error": "Invalid location."}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 400


@app.route('/get_insights', methods=['GET'])
def get_insights():
    # Calculate basic insights from sample data
    avg_waste = sample_waste_data['Waste Generated'].mean()
    max_waste = sample_waste_data['Waste Generated'].max()
    min_waste = sample_waste_data['Waste Generated'].min()

    # Render the insights in a template
    return render_template('get_insights.html', avg_waste=avg_waste, max_waste=max_waste, min_waste=min_waste)



@app.route('/bins', methods=['GET'])
def bins():
    return render_template('bin_data.html')


@app.route('/get_bin_data', methods=['POST'])
def get_bin_data():
    bin_name = request.form.get("bin")  # Retrieve the 'bin' field from the form
    if not bin_name:
        return jsonify({"error": "No bin selected."}), 400  # Handle missing bin selection

    if bin_name in bins_data:
        bin_info = bins_data[bin_name]
        filled = bin_info["filled"]
        capacity = bin_info["capacity"]
        empty_space = capacity - filled

        return jsonify({
            "bin": bin_name,
            "filled": f"{filled} kg",
            "empty": f"{empty_space} kg",
            "capacity": f"{capacity} kg"
        })
    else:
        return jsonify({"error": "Invalid bin selected."}), 400



@app.route('/maps', methods=['GET'])
def maps():
    return render_template('maps.html')


if __name__ == '__main__':
    app.run(debug=True)
