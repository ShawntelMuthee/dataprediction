import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Nairobi Office Price Ex.csv")
x = data['SIZE'].values  # Office sizes in square feet
y = data['PRICE'].values  # Office prices

# Function to compute Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function to update slope (m) and y-intercept (c)
def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * x + c
    dm = (-2 / n) * np.sum(x * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Training linear regression model using Gradient Descent
def train_linear_regression(x, y, learning_rate=0.0001, epochs=10):
    m, c = np.random.rand(), np.random.rand()
    errors = []
    for epoch in range(epochs):
        y_pred = m * x + c
        error = mean_squared_error(y, y_pred)
        errors.append(error)
        print(f"Epoch {epoch + 1}: MSE = {error:.4f}")
        m, c = gradient_descent(x, y, m, c, learning_rate)
    return m, c, errors

# Train the model
m, c, errors = train_linear_regression(x, y, learning_rate=0.0001, epochs=10)

# Plot 1: MSE over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), errors, marker='o')
plt.title("Mean Squared Error Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
# Save the MSE plot
plt.savefig('mse_plot.png')
plt.show()

# Plot 2: Line of Best Fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + c, color='red', label='Line of Best Fit')
plt.title("Line of Best Fit after Training")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.legend()
plt.grid(True)
# Save the line of best fit plot
plt.savefig('best_fit_plot.png')
plt.show()

# Predict office price for 100 sq. ft.
size = 100
predicted_price = m * size + c
print(f"\nModel Parameters:")
print(f"Slope (m): {m:.4f}")
print(f"Y-intercept (c): {c:.4f}")
print(f"The predicted office price for 100 sq. ft. is: {predicted_price:.2f}")