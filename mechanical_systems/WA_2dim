import numpy as np

def get_derivative_weighted_average(newq, newp, start, delta, final):
  q,p,qbar,pbar,qprime,pprime = start[0,:], start[1,:], final[0,:], final[1,:], delta[0,:], delta[1,:]
  weight = 1/(np.subtract(newq,q) + np.subtract(newp,p))**2
  weight = np.divide(weight, np.sum(weight))
  estqprime, estpprime = np.sum(np.multiply(weight, qprime)), np.sum(np.multiply(weight, pprime))
  return estqprime, estpprime
  
def LeapfrogWA(z,h, start, delta, final):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values

	dim = int(len(z)/2)
	z[dim:] = z[dim:]+h/2*get_derivative_weighted_average(z[:dim][0],z[dim:][0], start, delta, final)[:dim][0]
	z[:dim] = z[:dim]+h*z[dim:]
	z[dim:] = z[dim:]+h/2*get_derivative_weighted_average(z[:dim][0],z[dim:][0], start, delta, final)[:dim][0]
	return z
  
def gen_one_trajWA(traj_len,startcon,h,start, delta, final,n_h = 800):
  h_gen = h/n_h
  x, final = startcon.copy(), startcon.copy()
  for i in range(traj_len):
    startcon=np.hstack((startcon,x))
    for j in range(0,int(n_h+1)):
      x=LeapfrogWA(x,h_gen,start, delta, final)
    final=np.hstack((final,x))
  return startcon[:,1:],final[:,1:]
