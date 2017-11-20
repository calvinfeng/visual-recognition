import numpy as np
import matplotlib.pyplot as plt
from simple_net import SimpleNetwork

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

if __name__ == "__main__":
  xtr, ytr = generate_data(1000)
  xte, yte = generate_data(100)
  input_dim, hidden_dim, output_dim = xtr.shape[1], 5, ytr.shape[1]

  #################################
  # Initialize the neural network #
  #################################
  network = SimpleNetwork(input_dim, hidden_dim, output_dim, std=1e-3)

  test_acc = (network.predict(xte) == np.argmax(yte, axis=1)).mean()
  print 'Test accuracy before training: %s' % str(test_acc)

  iters, loss_hist, acc_hist = network.train(xtr, ytr)

  test_acc = (network.predict(xte) == np.argmax(yte, axis=1)).mean()
  print 'Test accuracy after training: %s' % str(test_acc)

  #####################
  # Print the weights #
  #####################
  print network.params['W1']
  print network.params['W2']

  plt.subplot(2, 1, 1)
  plt.plot(iters, loss_hist)
  plt.title('Loss vs Iterations')

  plt.subplot(2, 1, 2)
  plt.plot(iters, acc_hist)
  plt.title('Accuracy vs Iterations')

  plt.show()
