import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load your Airbnb listings data
listings_df = pd.read_csv('listings.csv')

# Select features 
features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'latitude', 'longitude', 'minimum_nights', 'availability_365'
]

# Handle missing values if any
listings_df[features] = listings_df[features].fillna(0)

# Convert 'price' column to numeric
listings_df['price'] = pd.to_numeric(listings_df['price'].replace('[\$,]', '', regex=True), errors='coerce')

# Drop rows with missing 'price' values
listings_df = listings_df.dropna(subset=['price'])

# Define the target variable
target = 'price'

# Split the data into features (X) and target variable (y)
X = listings_df[features]
y = listings_df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize the predicted vs. actual prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Predicted vs. Actual Prices (Linear Regression)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
