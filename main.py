from get_data import get_data
import analysis_new import add_ones, split_data
import train_new import LinearRegression
from test import mse


# getting the dataset
X_data, Y_data = get_data()

# adding a new column to the features with 1's
new_X_data = add_ones(X_data)

# splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = split_data(new_X_data, Y_data, 0.8)

# Instantiate the LinearRegression class 
# You can choose between the Normal Equation method or the Gram Schmidt Method
# For the Normal Equation, type "Normal Method" in the argument of the class, and "Gram Schmidt" for the Gram Schmidt method.
model2= LinearRegression("Normal Equation")

# Train the model
model2.fit(X_train,y_train)

# Make a prediction on X_test
y_pred3 = model2.predict(X_test)

# Compute the MSE (Evaluate both, regression and classification)
mse(y_test, y_pred3)