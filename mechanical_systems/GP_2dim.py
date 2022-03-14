# GP takes in q and p and learns the derivative dq/dt, dp/dt

import numpy as np 

def LeapfrogGP(z,h,gaussian_process,scaler):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values
	dim = int(len(z)/2)
	z[dim:] = z[dim:]+h/2*gaussian_process.predict(scaler.transform(z.transpose()))[0][1]
	z[:dim] = z[:dim]+h*z[dim:]
	z[dim:] = z[dim:]+h/2*gaussian_process.predict(scaler.transform(z.transpose()))[0][1]
	return z
  
def gen_one_trajGP(traj_len,start,h,gaussian_process,scaler,n_h = 800):
  h_gen = h/n_h
  x, final = start.copy(), start.copy()
  for i in range(traj_len):
    start=np.hstack((start,x))
    for j in range(0,int(n_h+1)):
      x=LeapfrogGP(x,h_gen,gaussian_process,scaler)
    final=np.hstack((final,x))
  return start[:,1:],final[:,1:]
