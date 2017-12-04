import numpy as np
import matplotlib.pyplot as plt
from sq_error_loss_net import SquaredErrorLossNetwork
from pdb import set_trace

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

######################################
# Generate some real legitimate data #
######################################
def generate_data(N):
    x, y = [], []
    for i in range(N):
        # Generate input vector
        randx = normalize(np.random.uniform(low=-1, high=1, size=2)) # Random normalized nearest obj position vector
        randv = normalize(np.random.uniform(low=-1, high=1, size=2)) # Random normalized current velocity vector
        randc = np.random.random_integers(0, high=1, size=1) # Random classification, either 0 or 1
        x.append(np.concatenate((randx, randv, randc)))

        # Calculate output vector
        cross_product = np.cross(randv, randx)

        # If it is a bacteria
        if randc == 1:
            # Bacteria is on the left
            if cross_product > 0:
                y.append([1, 0, 0])
            # Bacteria is on the right
            elif cross_product < 0:
                y.append([0, 1, 0])
            # Bacteria is in the line of sight
            else:
                if np.allclose(randx, randv, atol=1e-2):
                    y.append([0, 0, 1])
                else:
                    y.append([1, 0, 0])
        else:
            # Red blood cell is on the left
            if cross_product > 0:
                y.append([0, 1, 0])
            # Red blood cell is on the right
            elif cross_product < 0:
                y.append([1, 0, 0])  
            # Red blood cell is in the line of sight
            else:
                if np.allclose(randx, randv, atol=1e-2):
                    y.append([1, 0, 0])
                else:
                    y.append([0, 0, 1])
      
    return np.array(x).astype(float), np.array(y).astype(float)

if __name__ == "__main__":
  xtr, ytr = generate_data(2000)
  xte, yte = generate_data(200)
  input_dim, hidden_dim, output_dim = xtr.shape[1], 5, ytr.shape[1]

  #################################
  # Initialize the neural network #
  #################################
  network = SquaredErrorLossNetwork(input_dim, hidden_dim, output_dim, std=1e-4)

  test_acc = (network.predict(xte) == np.argmax(yte, axis=1)).mean()
  print 'Test accuracy before training: %s' % str(test_acc)

  iters, loss_hist, acc_hist = network.train(xtr, ytr, learning_rate=5e-2)

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
