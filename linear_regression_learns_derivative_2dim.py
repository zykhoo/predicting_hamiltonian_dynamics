# from sklearn.linear_model import LinearRegression
import numpy as np

# fit a linear regression model using sklearn, then pass it into this function

# q,p,qbar,pbar,qprime,pprime = start[0,:], start[1,:], final[0,:], final[1,:], delta[0,:], delta[1,:]
# X = start.transpose()
# # X = np.hstack((X, np.sin(X))) # this can drastically improve results, but we dont have this knowledge
# yreg = (final-start)/h
# yreg = yreg.transpose()
# # regq = LinearRegression().fit(X, y[:,0])
# regp = LinearRegression().fit(X, yreg[:,1])
# regp.predict([[0.4,0]]), regp.score(X,yreg[:,1])

def LeapfrogREG(z,h,reg):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values
	dim = int(len(z)/2)
	z[dim:] = z[dim:]+h/2*reg.predict(z.transpose())[0]
	z[:dim] = z[:dim]+h*z[dim:]
	z[dim:] = z[dim:]+h/2*reg.predict(z.transpose())[0]
	return z
  
def gen_one_trajREG(traj_len,start,h,reg,n_h = 800):
  h_gen = h/n_h
  x, final = start.copy(), start.copy()
  for i in range(traj_len):
    start=np.hstack((start,x))
    for j in range(0,int(n_h+1)):
      x=LeapfrogREG(x,h_gen,reg)
    final=np.hstack((final,x))
  return start[:,1:],final[:,1:]