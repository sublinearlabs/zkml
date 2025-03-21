import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from skl2onnx import to_onnx

# Load the Boston Housing dataset
boston = datasets.get_data_home()
X = boston.data
y = boston.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the range of alpha values to test
alphas = np.logspace(-4, 4, 10)

# Create a RidgeCV object with the specified alphas and 5-fold cross-validation
ridge = RidgeCV(alphas=alphas, cv=5)

# Fit the model to the training data
ridge.fit(X_train, y_train)

# Get the best alpha value
best_alpha = ridge.alpha_
print("Best alpha:", best_alpha)

# Make predictions on the test set
y_pred = ridge.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)



onx = to_onnx(ridge, X_train[:1])
with open("ridge_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
