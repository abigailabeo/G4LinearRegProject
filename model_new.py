import numpy as np

# cgs
def cgs(A):
  """
    Q,R = cgs(A)
    Apply classical Gram-Schmidt to mxn rectangular/square matrix. 

    Parameters
    -------
    A: mxn rectangular/square matrix   

    Returns
    -------
    Q: mxn square matrix
    R: nxn upper triangular matrix

  """
  # ADD YOUR CODES
  m = A.shape[0]  # get the number of rows of A
  n = A.shape[1] # get the number of columns of A

  R = np.zeros((n,n)) # create a zero matrix of nxn
  Q = np.ones((m,n)) # copy A (deep copy)


  for k in range(0,n):
    w = A[:,k]
    #print(w)
    
    for j in range(k-1):
      R[j,k] = Q[:,j].T@w
    
    for j in range(k-1):
      w = w - R[j,k]*Q[:,j]
    
    #norm = np.linalg.norm(w)
    
    R[k,k] = np.linalg.norm(w)
    Q[:,k] = w/R[k,k]
    #print(w/R[k,k])

  return  Q, R



# Implement BACK SUBS
def backsubs(U, b):

  """
  x = backsubs(U, b)
  Apply back substitution for the square upper triangular system Ux=b. 

  Parameters
  -------
    U: nxn square upper triangular array
    b: n array
    

  Returns
  -------
    x: n array
  """

  n= U.shape[1]
  x= np.zeros((n,))
  b_copy= np.copy(b)

  if U[n-1,n-1]==0.0:
    if b[n-1] != 0.0:
      print("System has no solution.")
  
  else:
    x[n-1]= b_copy[n-1]/U[n-1,n-1]
  for i in range(n-2,-1,-1):
    if U[i,i]==0.0:
      if b[i]!= 0.0:
        print("System has no solution.")
    else:
      for j in range(i,n):
        b_copy[i] -=U[i,j]*x[j]
      x[i]= b_copy[i]/U[i,i]
  return x

def normalEquation(X,y):
    theta = np.dot(np.linalg.inv(X.T@X),np.dot(X.T,y))
    return theta






