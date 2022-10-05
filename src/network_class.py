import numpy as np
import random

def activation(act_name, z):
    if act_name == "relu":
        return np.maximum(0, z)
    if act_name == "tanh":
        return np.tanh(z)                
    if act_name == "sigmoid":
        return 1.0/(1.0+np.exp(-z))

def activation_prime(act_name, z):
    if act_name == "relu":
        return np.where(z > 0, 1.0, 0.0)
    if act_name == "tanh":
        return 1 - np.tanh(z)**2
    if act_name == "sigmoid":
        return activation('sigmoid', z)*(1-activation('sigmoid', z))

class QuadraticCost():
     
    @staticmethod   
    def fn(a, y):
        return np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z, a, y, act_name):
        return (a-y) * activation_prime(act_name, z)


class CrossEntropyCost():

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class Network():

    def __init__(self, sizes, activations, b=0, w=0, cost = QuadraticCost):       
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activations = activations
        self.cost = cost
        
        #initializing the network with random weights and biases
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
         
        #providing the network with weights and biases values
        if b != 0:
            for i in range(self.num_layers-1):
                self.biases[i] = b[i]
        if w != 0:
            for i in range(self.num_layers-1):
                self.weights[i] = w[i]
                                
     
    #function to train the network using stohastic gradient descent
    #the data are a lists of tuples (x,y), where x and y are numpy 2D arrays
    def SGD(self, training_data, epochs, eta, mini_batch_size, test_data=False, dropout=False, trace_cost=False):
        
        self.n = len(training_data)

        for epoch, eta in zip(epochs, eta):
            for ep in range(epoch):               
                
                #separating the training data into mini-baches
                np.random.shuffle(training_data)
                mini_batches = [training_data[k:k+ mini_batch_size] for k in range(0, self.n, mini_batch_size )]
                
                #called if dropout method is used
                if dropout:                        
                    #dictionares where the position and the value of the weights and the biases, which will be erized, will be stored
                    bias_dict = {}
                    weight_rows_dict = {}
                    weight_columns_dict = {}
                    
                    for i in range(self.num_layers-2):
                        bias_dict[i] = {}    
                        weight_rows_dict[i] = {}
                        weight_columns_dict[i+1] = {}  
                        
                        hidden_neurons = sorted(random.sample(range(0, len(self.biases[i])), int(dropout*len(self.biases[i]))))                    
                        
                        for j in hidden_neurons:
                            bias_dict[i][j] = self.biases[i][[j]]
                            weight_rows_dict[i][j] = self.weights[i][[j]]
                            weight_columns_dict[i+1][j] = [row[j] for row in self.weights[i+1]]
    
                        #changing the shape of the weights and biases arrays, by delliting neurons 
                        self.biases[i] = np.delete(self.biases[i], hidden_neurons, 0)
                        self.weights[i] = np.delete(self.weights[i], hidden_neurons, 0)
                        self.weights[i+1] = np.delete(self.weights[i+1], np.s_[hidden_neurons], 1)
             
                        #calling method to update the weights and biases
                        for mini_batch in mini_batches:
                        self.update_mini_batch(mini_batch, eta)
                    
                    #restoring the initial biases and weights array's shape
                    for r_name, r_info in bias_dict.items():
                        for key in r_info:
                            self.biases[r_name] = np.insert(self.biases[r_name], key, r_info[key], axis =0)
                    for r_name, r_info in weight_rows_dict.items():
                        for key in r_info:
                            self.weights[r_name] = np.insert(self.weights[r_name], key, r_info[key], axis =0)
                    for r_name, r_info in weight_columns_dict.items():
                        for key in r_info:
                            self.weights[r_name] = np.insert(self.weights[r_name], key, r_info[key], axis =1)                           
                
                #updating weights and biases without dropout
                else:
                    for mini_batch in mini_batches:
                        self.update_mini_batch(mini_batch, eta)
                
                if test_data != False:
                    n_test = len(test_data)
                    print("Epoch {0}: {1} / {2}".format(ep, self.evaluate(test_data), n_test))                

                if trace_cost != False:
                    cost = 0
                    for x,y in training_data:
                        cost += (self.cost).fn(self.feedforward(x), y)
                    cost = cost/self.n
                    print("cost = ", cost)
       
                print("Epoch %s training complete" % ep)
                 
        #saving weights and biases arrays when training is finished
        self.weights = np.asanyarray(self.weights, dtype=object)       
        self.biases = np.asanyarray(self.biases, dtype=object)
        np.save('weights.npy', self.weights)
        np.save('biases.npy', self.biases)

    
    def update_mini_batch(self, mini_batch, eta):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        #rule that updates weights and biases
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
 
    
    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #feedforward pass
        current_activation = x
        Activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w, act in zip(self.biases, self.weights, self.activations):
            z = np.dot(w, current_activation)+b
            zs.append(z)
            current_activation = activation(act, z)
            Activations.append(current_activation)
        
        #backward pass 
        #computing delta for the output layer
        delta = (self.cost).delta(zs[-1], Activations[-1], y, act)
        #delta = (self.cost).delta(zs[-1], Activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, Activations[-2].transpose())
        
        #computing delta for the other layers in reverse order
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = activation_prime(self.activations[-l], z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, Activations[-l-1].transpose())
        return (nabla_b, nabla_w)
            
    def feedforward(self, a):
        for b, w, act in zip(self.biases, self.weights, self.activations):
            a = activation(act, np.dot(w, a)+b)
        return a
    
    def accuracy(self, validation_data):
        n = len(validation_data)
        e = 0.1
        acc = 0
        for x, y in validation_data:
            thres = (len(x)*95.0)/100
            output = self.feedforward(x) 
            count = 0
            for xx, yy in zip(output, y):
                if abs(xx - yy) < e:
                    count += 1
            if count > thres:
                acc += 1
              print( acc, "/", n)
        
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        ret = sum(1 for (x, y) in test_results if x==y)
        return ret
    
    
    