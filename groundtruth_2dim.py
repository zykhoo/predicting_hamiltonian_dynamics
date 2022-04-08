from tqdm import tqdm
import numpy as np
from skopt.space import Space
from skopt.sampler import Halton

def classicInt(z,f1,f2,h):
	## classical symplectic Euler scheme
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]
		fstage = lambda stg: h * f1(np.block([q + stg, p]))

		stageold=np.zeros(dim) 
		stage = fstage(stageold) +0.
		Iter = 0

		while (np.amax(abs(stage - stageold)) > 1e-10 and Iter<100):
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1
		q = q+stage
		p = p + h*f2(np.block([q,p]))
		return np.block([q,p])

def classicTrajectory(z,f1,f2,h,N=10,n_h=1):
	## trajectory computed with classicInt
  h_gen = h/n_h
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+1))
  trj[:,0] = z.copy()

  for i in range(0,N):
    for j in range(0,int(n_h+1)):
      trj[:,i+1] = classicInt(trj[:,i].copy(),f1,f2,h_gen)
  return trj[:, :-1], trj[:, 1:]


def CreateTrainingDataTrajClassicInt(traj_len,ini_con,spacedim,h,f1,f2,n_h = 800):
  space = Space(spacedim)
  h_gen = h/n_h
  halton = Halton()
  startcon = np.array(halton.generate(space, ini_con)).transpose()
  finalcon = startcon.copy()
  # Compute flow map from Halton sequence to generate learning data
  if ini_con==1: return classicTrajectory(startcon,f1,f2,h,N=traj_len)
  else:
    start, final= classicTrajectory(np.squeeze(startcon[:,0]),f1,f2,h,N=traj_len)
    for k in range(ini_con-1):
      new_start, new_final = classicTrajectory(np.squeeze(startcon[:,k+1]),f1,f2,h,N=traj_len)
      start = np.hstack((start, new_start))
      final = np.hstack((final, new_final))
  return start,final

def CreateTrainingDataTrajClassicIntRandom(traj_len,ini_con,spacedim,h,f1,f2,seed,n_h = 800):
  np.random.seed(seed = seed)
  startcon = np.random.uniform(spacedim[0][0], spacedim[0][1], size = ini_con)
  for i in range(len(spacedim)-1):
    startcon = np.vstack((startcon, np.random.uniform(spacedim[i+1][0], spacedim[i+1][1], size = ini_con)))
  h_gen = h/n_h
  finalcon = startcon.copy()
  # Compute flow map from Halton sequence to generate learning data
  if ini_con==1: return classicTrajectory(startcon,f1,f2,h,N=traj_len)
  else:
    start, final= classicTrajectory(np.squeeze(startcon[:,0]),f1,f2,h,N=traj_len)
    for k in range(ini_con-1):
      new_start, new_final = classicTrajectory(np.squeeze(startcon[:,k+1]),f1,f2,h,N=traj_len)
      start = np.hstack((start, new_start))
      final = np.hstack((final, new_final))
  return start,final

def get_within_array(trajectories, spacedim):
  within_array = np.asarray([])
  for i in range(len(trajectories)):
    np.sum(np.square(np.asarray([spacedim[0][0], spacedim[0][1]]), np.asarray([spacedim[1][0], spacedim[1][1]])))
    try:
      v = np.amin(np.concatenate((np.where(trajectories[i][1][0]<spacedim[0][0])[0],np.where(trajectories[i][1][0]>spacedim[0][1])[0], 
              np.where(trajectories[i][1][1]<spacedim[1][0])[0], np.where(trajectories[i][1][1]>spacedim[1][1])[0])))
      within_array = np.append(within_array, v)
    except ValueError:
      within_array = np.append(within_array, len(trajectories[i][1][1]))
  return within_array
