import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def get_data():
    # Set parameters
    np.random.seed(0)  # For reproducibility
    n_samples = 100    # Number of data points
    # Generate x values
    X = np.linspace(-5, 5, n_samples)
    # Define true polynomial coefficients for a quadratic polynomial
    true_coeffs = [1.0, -2.0, 1.0]  # Example coefficients for a quadratic polynomial
    # Compute y values using polynomial function
    y = np.polyval(true_coeffs, X)
    # Optionally, add some noise to the data
    noise = np.random.normal(0, 2.0, size=y.shape)
    y += noise

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    return [X, X_train, X_test, y_train, y_test]

def test(X, X_train, X_test, y_train, y_test, y_pred_curve):
    # Plot the data
    plt.figure(figsize=(12, 8))
    # Plot training data
    plt.scatter(X_train, y_train, color='blue', label='Training data', alpha=0.6)
    # Plot testing data
    plt.scatter(X_test, y_test, color='green', label='Testing data', alpha=0.6)
    # Plot predicted polynomial curve
    plt.plot(X, y_pred_curve, color="black", linewidth=2, label="Prediction")
    # Enhance plot
    plt.title('Polynomial Regression (Degree 2)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    # Show plot
    plt.show()
