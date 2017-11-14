import numpy as np

##########################
# TODO: Useful for later #
##########################
max_velocity = np.array([50, 50]).astype(float)
velocity_norm = np.linalg.norm(max_velocity)

max_spatial_dim = np.array([500, 500]).astype(float)
spatial_norm = np.linalg.norm(max_spatial_dim)

######################################
# Generate some real legitimate data #
######################################
def generate_data(N):
  x, y = [], []
  for i in range(N):
    # Generate input vector
    randx = np.random.uniform(low=-1, high=1, size=2) # Random nearest obj position vector
    randv = np.random.uniform(low=-1, high=1, size=2) # Random current velocity vector
    randc = np.random.random_integers(0, high=1, size=1) # Random classification, either 0 or 1
    x.append(np.concatenate((randx, randv, randc)))

    # Calculate output vector
    direc = randx - randv

    # If it is NOT a bacteria, move away from it
    if randc == 0:
      direc = -1 * direc

    idx = np.argmax(np.abs(direc))
    if idx == 0:
      # Move in the x direction
      if direc[0] < 0:
        y.append([1, 0, 0, 0])
      else:
        y.append([0, 1, 0, 0])
    else:
      # Move in the y direction
      if direc[1] < 0:
        y.append([0, 0, 1, 0])
      else:
        y.append([0, 0, 0, 1])

  return np.array(x).astype(float), np.array(y).astype(float)


xtr, ytr = generate_data(1)
print xtr
print ytr
