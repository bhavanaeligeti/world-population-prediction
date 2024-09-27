import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample historical data (replace with actual data)
data = {
    "Year": [1960, 1970, 1980, 1990, 2000, 2010, 2020],
    "Population": [3.03, 3.68, 4.43, 5.32, 6.15, 6.92, 7.76]  # In billions
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Reshape the data for sklearn (requires 2D array for features)
X = df[['Year']].values  # Feature: Year
y = df['Population'].values  # Target: Population

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict population for future years (e.g., 2025, 2030, 2040)
future_years = np.array([[2025], [2030], [2040]])
predicted_population = model.predict(future_years)

# Plotting the historical and predicted population
plt.scatter(X, y, color='blue', label='Historical Data')
plt.plot(future_years, predicted_population, color='red', linestyle='--', label='Predicted Population')
plt.xlabel('Year')
plt.ylabel('Population (Billions)')
plt.title('World Population Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Display predictions
for year, pop in zip(future_years.flatten(), predicted_population):
    print(f"Predicted Population in {year}: {pop:.2f} Billion")
