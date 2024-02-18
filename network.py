#Actualizacion 18 Feb CrossEntropy
#Este si es el chido bambi
import random 
import numpy as np 

class Network(object): 

    def __init__(self, sizes):
        self.num_layers = len(sizes) 
        self.sizes = sizes  
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.m_w = [np.zeros(w.shape) for w in self.weights]  # Momento para los pesos
        self.m_b = [np.zeros(b.shape) for b in self.biases]  # Momento para los biases
        self.v_w = [np.zeros(w.shape) for w in self.weights]  # Segundo momento para los pesos
        self.v_b = [np.zeros(b.shape) for b in self.biases]  # Segundo momento para los biases
        self.beta1 = 0.9  # Factor de decaimiento exponencial para el momento
        self.beta2 = 0.999  # Factor de decaimiento exponencial para el segundo momento
        self.epsilon = 1e-8  # Pequeña cantidad para evitar la división por cero en Adam

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: 
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)  
        n = len(training_data)
        for j in range(epochs): 
            random.shuffle(training_data) 
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] 
            for mini_batch in mini_batches:    
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j)) 

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        for x, y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        m_w_correction = [m_w / (1 - self.beta1) for m_w in self.m_w]
        m_b_correction = [m_b / (1 - self.beta1) for m_b in self.m_b]
        v_w_correction = [v_w / (1 - self.beta2) for v_w in self.v_w]
        v_b_correction = [v_b / (1 - self.beta2) for v_b in self.v_b]
        self.weights = [w - (eta / len(mini_batch)) * (m_w_corr / (np.sqrt(v_w_corr) + self.epsilon))
                        for w, m_w_corr, v_w_corr in zip(self.weights, m_w_correction, v_w_correction)]
        self.biases = [b - (eta / len(mini_batch)) * (m_b_corr / (np.sqrt(v_b_corr) + self.epsilon))
                       for b, m_b_corr, v_b_corr in zip(self.biases, m_b_correction, v_b_correction)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        # feedforward
        activation = x 
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b 
            zs.append(z) 
            activation = sigmoid(z) 
            activations.append(activation) 
        # backward pass
        delta = self.CrossEntropy_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers): 
            z = zs[-l]  
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        # Actualizar los momentos
        self.m_w = [(self.beta1 * m_w) + ((1 - self.beta1) * nw) for m_w, nw in zip(self.m_w, nabla_w)]
        self.m_b = [(self.beta1 * m_b) + ((1 - self.beta1) * nb) for m_b, nb in zip(self.m_b, nabla_b)]
        self.v_w = [(self.beta2 * v_w) + ((1 - self.beta2) * np.square(nw)) for v_w, nw in zip(self.v_w, nabla_w)]
        self.v_b = [(self.beta2 * v_b) + ((1 - self.beta2) * np.square(nb)) for v_b, nb in zip(self.v_b, nabla_b)]
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def CrossEntropy_derivative(self, output_activations, y):
        return ((output_activations - y) / (output_activations * (1 - output_activations)))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))