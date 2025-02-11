from flask import Flask, request, jsonify, render_template
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Initialize Nominatim geolocator
geolocator = Nominatim(user_agent="nyc_taxi_fare_app")

# Function to geocode an address using Nominatim
def geocode(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except GeocoderTimedOut:
        return None, None
    return None, None

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in a 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Geocode the pickup and dropoff addresses
    pickup_lat, pickup_lng = geocode(data['pickup_address'])
    dropoff_lat, dropoff_lng = geocode(data['dropoff_address'])

    # Validate if locations were found
    if None in [pickup_lat, pickup_lng]:
        return jsonify({'error': f"Pickup address '{data['pickup_address']}' not found"}), 400
    if None in [dropoff_lat, dropoff_lng]:
        return jsonify({'error': f"Dropoff address '{data['dropoff_address']}' not found"}), 400

    # Prepare features for prediction
    features = np.array([[pickup_lng, pickup_lat, dropoff_lng, dropoff_lat, data['passenger_count']]])
    prediction = model.predict(features)

    return jsonify({'fare_amount': round(prediction[0], 2)})

if __name__ == "__main__":
    app.run(debug=True)
