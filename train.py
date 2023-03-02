import numpy as np
class LinearRegression:

  def __init__(self,arg):
    self.theta1 = 0
    self.arg = arg
      
  def fit(self,x,y):
    if self.arg == "Normal Equation":
      self.theta = normalEquation(x,y)
    elif self.arg == "Gram Schmidt":
      Q1, R1 = cgs(x)
      self.theta  =backsubs(R1, np.dot(Q1.T,y))
    
  def predict(self,x):
    predicted_values = x@self.theta
    return predicted_values
  
#