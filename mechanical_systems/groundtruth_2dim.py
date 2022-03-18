import numpy as np 
# from skopt.space import Space
# from skopt.sampler import Halton

def Leapfrog(z,h,f):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values

	dim = int(len(z)/2)

	z[dim:] = z[dim:]+h/2*f(z[:dim])
	z[:dim] = z[:dim]+h*z[dim:]
	z[dim:] = z[dim:]+h/2*f(z[:dim])

	return z

def gen_one_traj(traj_len,start,h,f2,f1=None,n_h = 800):
  h_gen = h/n_h
  x, final = start.copy(), start.copy()
  for i in range(traj_len):
    start=np.hstack((start,x))
    for j in range(0,n_h+1):
      if f1 is None: 
        x=Leapfrog(x,h_gen,f2)
    final=np.hstack((final,x))
  return start[:,1:],final[:,1:]

def CreateTrainingDataTrajLeapfrog(traj_len,ini_con,spacedim,h,forces,n_h = 800):
  space = Space(spacedim)
  h_gen = h/n_h
  halton = Halton()
  startcon = np.array(halton.generate(space, ini_con)).transpose()
  finalcon = startcon.copy()
  # Compute flow map from Halton sequence to generate learning data
  if ini_con==1: return gen_one_traj(traj_len,startcon,h,forces)
  else:
    start,final=gen_one_traj(traj_len,startcon[:,0].reshape(-1,1),h,forces)
    for k in range(ini_con-1):
      new_start, new_final = gen_one_traj(traj_len,startcon[:,k+1].reshape(-1,1),h,forces)
      start = np.hstack((start, new_start))
      final = np.hstack((final, new_final))
  return start,final

def CreateTrainingDataTrajLeapfrogRandom(traj_len,ini_con,spacedim,h,forces,seed,n_h = 800):
  np.random.seed(seed = seed)
  startcon = np.random.uniform(spacedim[0][0], spacedim[0][1], size = ini_con)
  for i in range(len(spacedim)-1):
    startcon = np.vstack((startcon, np.random.uniform(spacedim[i+1][0], spacedim[i+1][1], size = ini_con)))
  h_gen = h/n_h
  finalcon = startcon.copy()
  # Compute flow map from Halton sequence to generate learning data
  if ini_con==1: return gen_one_traj(traj_len,startcon,h,forces)
  else:
    start,final=gen_one_traj(traj_len,startcon[:,0].reshape(-1,1),h,forces)
    for k in range(ini_con-1):
      new_start, new_final = gen_one_traj(traj_len,startcon[:,k+1].reshape(-1,1),h,forces)
      start = np.hstack((start, new_start))
      final = np.hstack((final, new_final))
  return start,final
