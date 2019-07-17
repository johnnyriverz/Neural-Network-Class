import numpy as np

class NeuralNetwork():

  def __init__ (self):
    np.random.seed()
    self.synaptic_weights = 2 * np.random.random((3,1)) - 1
  
  def sigmoid (self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative (self, x):
    return np.exp(-x) / (pow((1 + np.exp(-x)),2))

  def train (self,training_input,training_output,training_iterations):
    outputs = self.think(training_input)
    error = training_output - outputs
    adjustments = np.dot(training_input.T, error * self.sigmoid_derivative(outputs))
    self.synaptic_weights += adjustments
  
  def think (self, inputs):
    inputs = inputs.astype(float)
    outputs = self.sigmoid(np.dot(inputs,self.synaptic_weights))

    return outputs

if __name__ == '__main__':
  neural_network = NeuralNetwork()

  print ("Random synaptic weights: ")
  print (neural_network.synaptic_weights)

  training_input = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

  training_output = np.array([[0,1,1,0]]).T

  neural_network.train(training_input,training_output,30000)

  print ("Synaptic weights after training: ")
  print (neural_network.synaptic_weights)

  A = str(input("Input 1: "))
  B = str(input("Input 2: "))
  C = str(input("Input 3: "))

  print ("New Situation: Input Data = "+ A, B, C)
  print ("Output data: ")
  print (neural_network.think(np.array([A,B,C])))
