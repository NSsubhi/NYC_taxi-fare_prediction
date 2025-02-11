import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('train.csv', nrows=50000)  # Load a subset for faster processing

# Data preprocessing
data = data.dropna()
data = data[(data['fare_amount'] > 0) & (data['fare_amount'] < 100)]
data = data[(data['pickup_longitude'] > -75) & (data['pickup_longitude'] < -72)]
data = data[(data['pickup_latitude'] > 40) & (data['pickup_latitude'] < 42)]
data = data[(data['dropoff_longitude'] > -75) & (data['dropoff_longitude'] < -72)]
data = data[(data['dropoff_latitude'] > 40) & (data['dropoff_latitude'] < 42)]

# Feature selection
features = data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
target = data['fare_amount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Save the processed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
