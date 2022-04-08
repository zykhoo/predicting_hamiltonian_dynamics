# GP takes in q and p and learns the derivative dq/dt, dp/dt
from ..metrics import MSE
from tqdm import tqdm
import time

import numpy as np 

def naiveIntGP(z,h,gaussian_process,scaler):
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]		
		dH = gaussian_process.predict(scaler.transform(z.transpose()))[0]
		q = q+h*dH[0]
		p = p + h*dH[1]
		return np.block([q,p])

def naiveTrajectoryGP(z,h,gaussian_process,scaler,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+1))
  trj[:,0] = z.copy()
  for j in range(0,N+1):
    trj[:,j+1] = naiveIntGP(trj[:,j].reshape(-1,1).copy(),h,gaussian_process,scaler)
  return trj[:, :-1], trj[:, 1:]

def classicIntGP(z,h,gaussian_process,scaler):
	## classical symplectic Euler scheme
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]
		fstage = lambda stg: h * gaussian_process.predict(scaler.transform(np.concatenate([q,p]).transpose()))[0][0]

		stageold=np.zeros(dim) 
		stage = fstage(stageold) +0.
		Iter = 0

		while (np.amax(abs(stage - stageold)) > 1e-8 and Iter<100):
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1
		q = q+stage
		p = p + h*gaussian_process.predict(scaler.transform(np.concatenate([q,p]).transpose()))[0][1]
		return np.block([q,p])

def classicTrajectoryGP(z,h,gaussian_process,scaler,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+1))
  trj[:,0] = z.copy()
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

def compute_metrics_GP(gp, scaler, h, diagdist, xshort, yshort, xlong, ylong, eval_len, len_within, long_groundtruth, len_short, truevector):
    results_start = np.asarray(classicTrajectoryGP(np.asarray([[0.4],[0.]]),h,gp,scaler,N=eval_len))
    withinspace_longtraj_symplectic_MSe = MSE(long_groundtruth[0,1,:,:], results_start[1,:,:], diagdist)
    results_start = np.asarray(naiveTrajectoryGP(np.asarray([[0.4],[0.]]),h,gp, scaler,N=eval_len))
    withinspace_longtraj_naive_MSe = MSE(long_groundtruth[0,1,:,:], results_start[1,:,:], diagdist)

    MSE_long, time_long, MSE_long_naive, time_long_naive, MSE_within, time_within, MSE_within_naive, time_within_naive, MSE_onestep, time_onestep, MSE_vectorfield, time_vectorfield = 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.
    count = 1
    for i in tqdm(np.expand_dims(np.c_[np.ravel(xlong),np.ravel(ylong)],2)):
      starttime = time.time()
      results_start = np.asarray(classicTrajectoryGP(i,h,gp,scaler,N=eval_len))
      time_long += time.time()-starttime
      MSE_long += MSE(long_groundtruth[count,1,:,:], results_start[1,:,:], diagdist)
      starttime = time.time()
      results_start = np.asarray(naiveTrajectoryGP(i,h,gp, scaler,N=eval_len))
      time_long_naive += time.time()-starttime
      MSE_long_naive += MSE(long_groundtruth[count,1,:,:], results_start[1,:,:], diagdist)
      steps = int(len_within[count-1])
      supp = (len_within>0).sum()
      if steps == 0:
        pass
      else: 
        starttime = time.time()
        results_start = np.asarray(classicTrajectoryGP(i,h,gp,scaler,N=steps-1))
        time_within += time.time()-starttime
        MSE_within += MSE(long_groundtruth[count,1,:,:steps], results_start[1,:,:], diagdist)
        starttime = time.time()
        results_start = np.asarray(naiveTrajectoryGP(i,h,gp, scaler,N=steps-1))
        time_within_naive += time.time()-starttime
        MSE_within_naive += MSE(long_groundtruth[count,1,:,:steps], results_start[1,:,:], diagdist)
      count+=1 
    count = 1
    for i in tqdm(np.expand_dims(np.c_[np.ravel(xshort),np.ravel(yshort)],2)):
      starttime = time.time()
      results_start = np.asarray(classicTrajectoryGP(i,h,gp,scaler,N=1))
      time_onestep += time.time()-starttime
      MSE_onestep += MSE(len_short[count,1,:,:], results_start[1,:,:], diagdist)
      starttime = time.time()
      vectorfield = gp.predict(scaler.transform(i.transpose()))[0].flatten()
      time_vectorfield += time.time()-starttime
      MSE_vectorfield += MSE(truevector(len_short[count,0,:,:].flatten()), vectorfield, diagdist)
      count+=1
    return MSE_long/25, time_long, MSE_long_naive/25, time_long_naive, MSE_within/supp, time_within, MSE_within_naive/supp, time_within_naive, MSE_onestep/400, time_onestep, MSE_vectorfield/400, time_vectorfield/400, withinspace_longtraj_symplectic_MSe, withinspace_longtraj_naive_MSe
