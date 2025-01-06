import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
file_path = "/content/auto-mpg.csv"
data = pd.read_csv(file_path)

# Step 2: Data preprocessing
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
data = data.dropna()

X = data["horsepower"].values.reshape(-1, 1)
y = data["mpg"].values

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print("Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 7: Visualize the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.title("Linear Regression: Horsepower vs MPG")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()





