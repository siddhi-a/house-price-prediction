import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("house_price_dataset.csv")

# Features and Target
X = df[['Area (sq ft)', 'Bedrooms', 'Age']]
y = df['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Show coefficients
print("\nModel Coefficients:")
print(f"Area Coefficient: {model.coef_[0]:.2f}")
print(f"Bedrooms Coefficient: {model.coef_[1]:.2f}")
print(f"Age Coefficient: {model.coef_[2]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Visualize Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
