from tqdm import tqdm
import numpy as np

def classicInt(z,f1,f2,h):
	## classical symplectic Euler scheme
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]
		fstage = lambda stg: h * f1(np.block([q + stg, p]))

		stageold=np.zeros(dim) 
		stage = fstage(stageold) +0.
		Iter = 0

		while (np.amax(abs(stage - stageold)) > 1e-8 and Iter<100):
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1
		q = q+stage
		p = p + h*f2(np.block([q,p]))
		return np.block([q,p])

def classicTrajectory(z,f1,f2,h,N=10,n_h=800):
	## trajectory computed with classicInt
  h_gen = h/n_h
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+2))
  trj[:,0] = z.copy()

  if N == 1:
    for j in range(0,int(n_h+1)):
      trj[:,0]=classicInt(trj[:,0],f1,f2,h_gen)
    return z.reshape(-1,1), trj[:,0].reshape(-1,1)
  else:
    for i in tqdm(range(0,N+1)):
      for j in range(0,int(n_h+1)):
        trj[:,i+1] = classicInt(trj[:,i].copy(),f1,f2,h_gen)
    return trj[:, :-1], trj[:, 1:]

