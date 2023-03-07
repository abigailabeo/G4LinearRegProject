def mse(y, y_pred):
    mean_sq_error = (1/len(y)) * np.sum((y-y_pred)**2)
    return mean_sq_error

def predict(self,x):
    predicted_values = x@self.theta
    return predicted_values

