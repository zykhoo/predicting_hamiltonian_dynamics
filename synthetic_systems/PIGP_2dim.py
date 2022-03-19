# adapted from Offen et al's Shadow Symplectic Integrator
# GP takes in p and q and learns the Hamiltonian
# fixed a bug in Offen's code (regarding the derivative of the kernel)

import numpy as np
import pickle
from datetime import datetime
from scipy.linalg import cho_factor, cho_solve
from tqdm import tqdm


def classicIntGPdH(z,h,GP):
	## classical symplectic Euler scheme
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]		
		fstage = lambda stg: h *GP.dH(z.transpose())[0]*(-1)

		stageold=np.zeros(dim) 
		stage = fstage(stageold) +0.
		Iter = 0

		while (np.amax(abs(stage - stageold)) > 1e-8 and Iter<100):
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1
		q = q+stage
		p = p + -h*GP.dH(z.transpose())[1]*(-1)
		return np.block([q,p])

def classicTrajectoryGPdH(z,h,GP,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+2))
  trj[:,0] = z.copy()
  if N == 1:
    return z.reshape(-1,1), classicIntGPdH(trj[:,0].reshape(-1,1),h,GP).reshape(-1,1)
  else:
    for j in range(0,N+1):
      trj[:,j+1] = classicIntGPdH(trj[:,j].reshape(-1,1).copy(),h,GP)
  return trj[:, :-1], trj[:, 1:]

def naiveIntGPdH(z,h,GP):
	## classical symplectic Euler scheme
		dim = int(len(z)/2)
		q=z[:dim]
		p=z[dim:]		
		q = q + h*GP.dH(z.transpose())[1]
		p = p + -h*GP.dH(z.transpose())[0]
		return np.block([q,p])

def naiveTrajectoryGPdH(z,h,GP,N=1):
	## trajectory computed with classicInt
  z = z.reshape(1,-1)[0]
  trj = np.zeros((len(z),N+2))
  trj[:,0] = z.copy()
  if N == 1:
    return z.reshape(-1,1), naiveIntGPdH(trj[:,0].reshape(-1,1),h,GP).reshape(-1,1)
  else:
    for j in range(0,N+1):
      trj[:,j+1] = naiveIntGPdH(trj[:,j].reshape(-1,1).copy(),h,GP)
  return trj[:, :-1], trj[:, 1:]

class BertalanGP():
	
	def classicInt(self,z,f1,f2,h,verbose = False):
	## classical symplectic Euler scheme
	
		dim = int(len(z)/2)
	
		q=z[:dim]
		p=z[dim:]

		fstage = lambda stg: h * f1(np.block([q + stg, p]))

		# fixed point iterations to compute stages

		stageold=np.zeros(dim)
		stage = fstage(stageold) +0.
		Iter = 0

		while (np.amax(abs(stage - stageold)) > 1e-8 and Iter<100):
			stageold = stage+0.
			stage = fstage(stage)+0.
			Iter = Iter+1

		if verbose == True:
			print('SEuler fixpoint iterations: ' + str(Iter) + ' Residuum: ' + str(abs(stage - stageold)))


		q = q+stage
		p = p + h*f2(np.block([q,p]))

		return np.block([q,p])



	def classicTrajectory(self,z,f1,f2,h,N=1,verbose=False, saveflag=False, saveint=1000):
	## trajectory computed with classicInt
	
		self.trj = np.zeros((len(z),N+1))
		self.trj[:,0] = z.copy()

		for j in tqdm(range(0,N)):
			self.trj[:,j+1] = self.classicInt(self.trj[:,j].copy(),f1,f2,h,verbose)
		
			if (saveflag==True and (j+1) % saveint == 0):
				timedata = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
				pickle.dump(self.trj[:,:j+1],open('SSIdata_'+str(j)+'_'+timedata+'.pkl','wb'))
		
		return self.trj
		
		
	def train(self,trainin, delta, h, k=False, dk=False, ddk=False, x0=0, H0=0, sigma=1e-13):
	# fit GP to training data 
	# with kernel k and derivatives dk, Hessian ddk (set as constant zero matrix if it does not exist)
	# if k is not specified, we use radial basis functions
	# normalisation H0 at x0
	# using Tikhonov regularisation with parameter sigma
	
	
		if k==False:
		# use radial basis functions if not specified otherwise
			
			# kernel function and its gradient wrt. one input
			e   = 2.           # length scale
			k   = lambda x,y: np.exp(-1/(e**2)*np.linalg.norm(x-y)**2)
			dk  = lambda x,y: -2/e**2*(x-y)*k(x,y)
			ddk = lambda x,y: 2/e**2*k(x,y)*(-np.identity(len(x)) + 2/e**2*np.outer(x-y,x-y))
		
	
		dim = int(len(trainin)/2)
	
		J=np.block([    [np.zeros((dim,dim)),-np.identity(dim)],    [np.identity(dim),np.zeros((dim,dim))] ])
		Jinv = -J

		# Inverse modified vector field for Symplectic Euler
		g=np.hstack((J @ delta).transpose())
	

		# data points for inference compatible with Symplectic Euler X=(Q,p)
		# X = [ np.block([trainin[:dim,j],trainin[dim:,j]]) for j in range(0,len(trainin[0])) ]
		# X = np.array(X)
		# print(X)
		X = trainin.transpose()


		# normalisation value H0 at H(x0)
		x0 = x0*np.ones(2*dim) # make sure, x0 has the correct size if it is not set explicitly
		RHS = np.append(g,H0)
	
		Y=X
	
		print("Start training with for "+str(len(Y))+ " data points")

		# Covariance matrix for the multivariate normal distribution of the random vector (H(Y[0]),H(Y[1]),...,H(Y[-1]))

		print("Start computation of covariance matrix.")
		K = [ [ k(Y[i],Y[j]) for i in range(0,Y.shape[0]) ] for j in range(0,Y.shape[0]) ]
		K = np.array(K)
		print("Covariance matrix of shape "+str(K.shape)+"computed.")

		# Tikhonov regularisation and Cholesky decomposition
		K = K + sigma*np.identity(K.shape[0])
		print("Start Cholesky decomposition of "+str(K.shape)+" Matrix")
		L, low = cho_factor(K)
		print("Cholesky decomposition completed.")

		# creation of linear systems to compute mean of inverse modified Hamiltonian from inverse modified vectorfield
		print("Create LHS of linear system for H at test points.")
		dkY = lambda x : [dk(x,y) for y in Y ]
		dKK = [cho_solve((L, low), np.array(dkY(xi))).transpose() for xi in X ]       # np.array(dkY(xi)).transpose() @ Kinv
		dKK=np.vstack(dKK)

		# normalisation of H
		kY = [k(x0,y) for y in Y]
		kYY=cho_solve((L, low), np.array(kY)).transpose()

		LHS = np.vstack([dKK,kYY])
		print("Creation of linear system completed.")

		# solution of linear system
		print("Solve least square problem of dimension "+str(LHS.shape))
		HY,res,rank, _ = np.linalg.lstsq(LHS,RHS,rcond=None)
		
		# provide data for other methods	
		self.h = h
		self.Jinv = Jinv
		self.k = k
		self.dk = dk
		self.ddk = ddk
		self.HX = HY
		self.X=X
		self.L = L
		self.KinvH = cho_solve((L,low),HY)

		return res, rank

	# mean values of inverse modified Hamiltonian and its jet at x infered from values at (X,HX)
	
	def H(self,x):

		kX = [self.k(x,y) for y in self.X ]
		return np.array(kX).transpose() @ self.KinvH  # H
	
	def dH(self,x):

		dkX = [self.dk(x,y) for y in self.X ]
		return np.array(dkX).transpose() @ self.KinvH  # grad H
	
	
	def ddH(self,x):

		ddkX = [self.ddk(x,y) for y in self.X ]

		return np.array(ddkX).transpose() @ self.KinvH  # Hessian H

	
	def predictMotion(self,z0,N,verbose=False,saveflag=False, saveint=1000):
	# apply classical integrator to inverse modified vector field

		dim = int(len(z0)/2)

		f1 = lambda z: (self.Jinv @ self.dH(z))[:dim]
		f2 = lambda z: (self.Jinv @ self.dH(z))[dim:]
		
		
		return self.classicTrajectory(z0,f1,f2,self.h,N,verbose=verbose,saveflag=saveflag, saveint=saveint)
  
	def gradient(self,z,verbose=False,saveflag=False, saveint=1000):
	# apply classical integrator to inverse modified vector field

		dim = int(len(z0)/2)

		f1 = h*(self.Jinv @ self.dH(z))[:dim]
		f2 = h*(self.Jinv @ self.dH(z))[dim:]
		
		
		return f1[0],f2[0]
		
		
	def HRecover(self,x):

		dim = int(len(x)/2)

		HH = self.H(x)
		grad = self.dH(x)
		hess = self.ddH(x)


		Hq = grad[:dim]
		Hp = grad[dim:]

		Hqq = hess[:dim,:dim]
		Hpp = hess[dim:,dim:]
		Hqp = hess[:dim,dim:]

		H1 = HH - self.h/2*(np.dot(Hq,Hp))
		H2 = H1 + self.h**2/12*( Hq.transpose() @ Hpp @ Hq  + Hp.transpose() @ Hqq @ Hp + 4*(Hp.transpose() @ Hqp @ Hq ))

		return HH,H1,H2
