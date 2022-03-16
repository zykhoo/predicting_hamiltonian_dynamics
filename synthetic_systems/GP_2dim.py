# GP takes in q and p and learns the derivative dq/dt, dp/dt

import numpy as np 

def classicIntGP(z,h,gaussian_process,scaler):
	## classical symplectic Euler scheme
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]		
		fstage = lambda stg: h * gaussian_process.predict(scaler.transform(z.transpose()))[0][0]

		stageold=np.zeros(dim) 
		stage = fstage(stageold) +0.
		Iter = 0

		while (np.amax(abs(stage - stageold)) > 1e-8 and Iter<100):
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1
		q = q+stage
		p = p + h*gaussian_process.predict(scaler.transform(z.transpose()))[0][1]
		return np.block([q,p])

def classicTrajectoryGP(z,h,gaussian_process,scaler,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+2))
  trj[:,0] = z.copy()
  if N == 1:
    return z.reshape(-1,1), classicIntGP(trj[:,0].reshape(-1,1),h,gaussian_process,scaler).reshape(-1,1)
  else:
    for j in range(0,N+1):
      trj[:,j+1] = classicIntGP(trj[:,j].reshape(-1,1).copy(),h,gaussian_process,scaler)
  return trj[:, :-1], trj[:, 1:]

def classicqbarpbar(z,h,gaussian_process,scaler):
		dim = int(len(z)/2)		
		qbar = gaussian_process.predict(scaler.transform(z.transpose()))[0][0]
		pbar = gaussian_process.predict(scaler.transform(z.transpose()))[0][1]
		return np.block([qbar,pbar])

def classicTrajectorybar(z,h,gaussian_process,scaler,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+2))
  trj[:,0] = z.copy()
  if N == 1:
    return z.reshape(-1,1), classicqbarpbar(trj[:,0].reshape(-1,1),h,gaussian_process,scaler).reshape(-1,1)
  else:
    for j in range(0,N+1):
      trj[:,j+1] = classicqbarpbar(trj[:,j].reshape(-1,1).copy(),h,gaussian_process,scaler)
  return trj[:, :-1], trj[:, 1:]
