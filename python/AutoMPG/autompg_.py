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

data.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
file_path = "/content/auto-mpg.csv"
data = pd.read_csv(file_path)

# Step 2: Data preprocessing
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
data = data.dropna()


X = data[["horsepower", "weight", "acceleration", "model year"]]  # Multiple features
y = data["mpg"]

# Step 3: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 8: Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_squared_error") 
print(f"Cross-validated Mean Squared Error: {-cv_scores.mean():.2f}")


# Step 9: Visualize the regression line (for a 2D feature visualization, we can plot 'horsepower' vs 'mpg')
plt.figure(figsize=(8, 6))
plt.scatter(data["horsepower"], y, color="blue", label="Data Points")
plt.plot(data["horsepower"], model.predict(scaler.transform(data[["horsepower", "weight", "acceleration", "model year"]])),
         color="red", label="Regression Line")
plt.title("Multiple Linear Regression: Horsepower, Weight, Acceleration, Year vs MPG")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()

data.head()





