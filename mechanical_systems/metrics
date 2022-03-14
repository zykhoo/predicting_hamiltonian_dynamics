import numpy as np

def MSE(arr, groundtruth, diagdist):
  assert arr.shape == groundtruth.shape, ("shape of array is", arr.shape, "shape of groundtruth is", groundtruth.shape)
  err = np.nan_to_num(((arr-groundtruth)**2), nan = diagdist)
  err[err>diagdist] = diagdist
  return np.mean(err)
