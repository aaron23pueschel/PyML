################################################################################

# Neural network class: FULLY CONNECTED NEURAL NETWORK 
# Written by: Aaron Pueschel, 12/20/2023 - 01/07/2024 
# 
# ------ Aspirations for this project -----
# This is the first iteration of neural network library that I hope to complete in the next coming
# Months. It is written with the same style as PyTorch, The idea is to create this library as a
# playground for various various projects, including CUDA HPC, code generation and others. I hope this 
# project will help me to learn the nuances of how a neural network is coded and trained.

# ----- Motivation -------
# This project began as an assignment for an optimization class I took in the fall of 2023. 
# The original intention was to create this class for that project in matlab, but I ended up
# doing it in pytorch. The idea was to explore how well various optimizing algorithms 
# do at learning noisy mappings for benchmark optimization functions on fully connected neural networks.
# This is a regression problem... relatively easy and not as interesting as classification problems
# I did both!


# ------ FEATURES ------
# Fully connected neural network
# Adam and SGD algorithm
# Various activation functions
# Classification and Regression capabilities
# Gradient clipping
# Tensor multiplication and sequential (More on this later)
# Validation metrics
# State save


###########################################################################


import numpy as np
import pickle
import os

class NN(object):
    def __init__(self,name = "Neural Net"):
        self.name = name


######################################################################
#               FULLY CONNECTED CLASS
######################################################################
        
# Much of this class will be moved into NN, when I decide what will be needed by 
# the convolution layers
class FullyConnected(NN):
    def __init__(self,inputDim,outputDim,NPL,HiddenDim,minibatch_sz, activation_fcn="SiLU",
                 cost_func="MSE", optimizer="SGD",classifier=False,dtype_ = np.float32, 
                 use_tensors=True,cost_function = "CrossEntropyLoss",output_bias=True,
                 input_bias=True):
        
        """
        Parameters:
        - inputDim (int): The dimensionality of the input data.
        - outputDim (int): The dimensionality of the output data.
        - NPL (int): Neurons per layer
        - HiddenDim (int): The number of hidden layer(s).
        - minibatch_sz (int): The size of the minibatch used in training.
        - activation_fcn (str): The activation function used in the network (default: "SiLU").
        - cost_func (str): The cost function used for training (default: "MSE").
        - optimizer (str): The optimizer used during training (default: "SGD").
        - classifier (bool): True if the network is a classifier, False otherwise (default: False).
        - dtype_ (numpy dtype): The data type used for computations (default: np.float32).
        - use_tensors (bool): True if tensors are used, False if not (default: True).
        - cost_function (str): The cost function used (default: "CrossEntropyLoss").
        - output_bias (bool): True if the output layer has bias, False otherwise (default: True).
        - input_bias (bool): True if the input layer has bias, False otherwise (default: True).  
        """

        # Initialize class variables
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.NPL = NPL
        self.HiddenDim = HiddenDim
        self.minibatch_sz = minibatch_sz
        self.activation_fcn=activation_fcn
        self.optimizer = optimizer
        self.classifier = classifier
        self.dtype_ = dtype_
        self.use_tensors = use_tensors
        self.output_bias = output_bias
        self.input_bias = input_bias

        self.parameters = dict()
        self.gradients = dict()
        self.init_weights()
        self.init_gradients()
        self.init_intermediate_vals()
        self.init_activationFunc(activation_fcn)
        self.init_alg_params()
        self.init_gradient_clipping()
        self.init_costFunc(cost_func)
        
        # Used for saving training/validation history
        self.loss_history = []
        self.train_validation =[]
        self.test_validation = []



    def init_gradients(self):
        """
        Initialize global gradients
        """
        self.gradients = {
            "weights_in": np.zeros((self.inputDim, self.NPL)),
            "weights": np.zeros((self.NPL, self.NPL, self.HiddenDim)),
            "weights_out": np.zeros((self.NPL, self.outputDim)),
            "bias": np.zeros((self.NPL, self.HiddenDim)),
            "output_bias": np.zeros((self.outputDim, 1)).transpose(),
            "input_bias": np.zeros((self.NPL, 1))
        }
        self.gradients_sample = {
            "weights_in": np.zeros((self.inputDim, self.NPL)),
            "weights": np.zeros((self.NPL, self.NPL, self.HiddenDim)),
            "weights_out": np.zeros((self.NPL, self.outputDim)),
            "bias": np.zeros((self.NPL, self.HiddenDim)),
            "output_bias": np.zeros((self.outputDim, 1)).transpose(),
            "input_bias": np.zeros((self.NPL, 1))
        }

    
    def init_weights(self):
        """
        Initialize global weights
        """
        self.parameters = {
            "weights_in": ((np.random.randn(self.inputDim, self.NPL) * np.sqrt(1.0 / self.inputDim)).T).astype(dtype=self.dtype_),
            "weights": np.random.randn(self.NPL, self.NPL, self.HiddenDim) * np.sqrt(2.0 / self.NPL),
            "weights_out": (np.random.randn(self.NPL, self.outputDim) * np.sqrt(1.0 / self.NPL)).T,
            "bias": np.zeros((self.NPL, self.HiddenDim)),
            "output_bias": np.zeros((self.outputDim, 1)),
            "input_bias": np.zeros((self.NPL, 1))
        }

    
    def init_intermediate_vals(self):
        """
        Initializes intermediate values used for backpropagation.
        """
        self.a = np.zeros((self.NPL, self.HiddenDim +1  ,self.minibatch_sz))
        self.z = np.zeros((self.NPL, self.HiddenDim +1,self.minibatch_sz))
        if self.classifier is True:
            self.z_out = np.zeros((self.outputDim,self.minibatch_sz))
    

    # Initializes global variables for optimizing algorithms
    def init_alg_params(self):
        """
        Initializes global variables for optimization algorithms.
        """
        if self.optimizer == "ADAM":
             self.adam_m = {param_name: np.zeros_like(param_value) for param_name, param_value in self.parameters.items()}
             self.adam_v = {param_name: np.zeros_like(param_value) for param_name, param_value in self.parameters.items()}
    
   
    def init_gradient_clipping(self):
        """
        Initializes global variables for gradient clipping, including running averages and running variance.

        Returns:
        None
        """
        self.n = 1
        self.mean = {
            "weights_in": 0,
            "weights_out": 0,
            "input_bias": 0,
            "output_bias": 0,
            "weights":np.zeros(self.HiddenDim),
            "bias":np.zeros(self.HiddenDim)
            }
        self.M2 = {
            "weights_in": 0,
            "weights_out": 0,
            "input_bias": 0,
            "output_bias": 0,
            "weights":np.zeros(self.HiddenDim),
            "bias":np.zeros(self.HiddenDim)
            }
      
    
    def init_activationFunc(self,activation_fcn):
        """
        Initializes the activation function for the hidden layers of the network (excluding the output layer).

        Parameters:
        - activation_fcn (str): The name of the activation function to be used.

        Returns:
        None
        """
        if activation_fcn == "ReLU":
            self.activationFunc = self.ReLU
        elif activation_fcn == "Tanh":
            self.activationFunc = self.tanh
        elif activation_fcn == "Sigmoid":
            self.activationFunc = self.sigmoid
        elif activation_fcn == "SiLU":
            self.activationFunc = self.silu
        elif self.activation_fcn == "Softmax":
            self.activationFunc = self.softmax
        elif self.activation_fcn == "GeLU":
            self.activationFunc = self.gelu
        else:
            raise RuntimeError("Unspecified activation function")
    


    def init_costFunc(self,loss_function):
        """
        Initializes the cost function for the model.

        Parameters:
        - loss_function (str): The name of the loss function to be used.

        Returns:
        None
        """
        if loss_function=="CrossEntropyLoss":
            self.costFunc = self.cross_entropy_loss
        if loss_function=="MSE":
            self.costFunc = self.MSE

    def forward_sample(self,x_initial,sample_no): 
        """
        Performs a forward pass computation for a specific input sample.

        Parameters:
        - x_initial (numpy array): The initial input data for the sample.
        - sample_no (int): The index or identifier of the sample.

        Returns:
        numpy array: The output of the forward pass for the specified sample.
        """
        x_initial = x_initial.reshape(self.inputDim,1)
        
        self.z[:,0:1,sample_no] = self.parameters["weights_in"] @ x_initial #+ self.parameters["input_bias"]
        self.a[:,0:1,sample_no] = self.activationFunc(self.z[:,0:1,sample_no],derivative=False)
        for i in range(self.HiddenDim):
            self.z[:,i+1,sample_no] = self.parameters["weights"][:, :, i] @ self.a[:, i,sample_no] #+self.parameters["bias"][:, i-1]
            self.a[:,i+1,sample_no] = self.activationFunc(self.z[:,i+1,sample_no],derivative=False)
        out = self.parameters["weights_out"] @ self.a[:,self.HiddenDim,sample_no].reshape(self.NPL,1) #+self.parameters["output_bias"]
        if not self.classifier:
            return out
        self.z_out[:,sample_no] = out.squeeze()
        return self.softmax(out.T,derivative=False).T



    def forward(self,x_minibatch):
        """
        Performs a forward pass computation for a given minibatch of input data.

        Parameters:
        - x_minibatch (numpy array): The input minibatch data.

        Returns:
        numpy array: The output of the forward pass.
        """
        if self.use_tensors:
            return self.forward_tensors(x_minibatch)
        y_out = np.zeros((self.minibatch_sz,self.outputDim))
        for i,x in enumerate(x_minibatch):
            y_out[i,:] = self.forward_sample(x,sample_no=i).flatten()
        return y_out
    


    def forward_tensors(self,x_minibatch):

        """
        Performs forward pass computation for a given minibatch of input data using numpy tensors

        Parameters:
        - x_minibatch (numpy array or tensor): The input minibatch data.

        Returns:
        numpy array: The output of the forward pass.
        """

        self.z[:,0:1,:] = np.tensordot(self.parameters["weights_in"][:,np.newaxis,:], x_minibatch.T ,axes=1) #+ self.parameters["input_bias"][:,np.newaxis]
        self.a[:,0:1,:] = self.activationFunc(self.z[:,0:1,:],derivative=False)

        for i in range(self.HiddenDim):
            self.z[:,i+1,:] = np.tensordot(self.parameters["weights"][:, :, i], self.a[:, i,:],axes=1) #+self.parameters["bias"][:, i-1,np.newaxis]
            self.a[:,i+1,:] = self.activationFunc(self.z[:,i+1,:],derivative=False)
        self.z_out = np.tensordot(self.parameters["weights_out"][:,np.newaxis,:], self.a[:,self.HiddenDim,:],axes=1) #+self.parameters["output_bias"][:,np.newaxis]
        self.z_out = self.z_out.squeeze()
        
        if not self.classifier:
            return self.activationFunc(self.z_out,derivative=False)
        return self.softmax(self.z_out.T,derivative=False).T
        


    def clip_gradients(self, threshold=3,clip_start=3000,print_if_clipped= False):

        """
        Clips gradients based on their norm within a specified threshold.

        Parameters:
        - threshold (float): The number of standard deviations from the mean for gradient clipping (default: 3).
        - clip_start (int): The number of samples needed before applying gradient clipping (default: 3000).
        - print_if_clipped (bool): Whether to print a message if gradients are clipped (default: False).

        Returns:
        None
        """
        
        if self.n <clip_start:
            return
        for key in self.gradients.keys():
            print_flag = False
            if key !="bias" and key !="weights":
                norm = np.linalg.norm(self.gradients[key])

                if norm > self.get_clip_mean(key) + threshold* self.get_clip_stddev(key):
                    self.gradients[key] = self.get_clip_mean(key)*(self.gradients[key]/norm)
                    print_flag = True
            elif key=="bias":
                for i in range(self.HiddenDim):
                    norm = np.linalg.norm(self.gradients[key][:,i])
                    if norm > self.get_clip_mean(key)[i] + threshold* self.get_clip_stddev(key)[i]:
                        self.gradients[key][:,i] = self.get_clip_mean(key)[i]*(self.gradients[key][:,i]/norm)
                        print_flag = True
            else:
                for i in range(self.HiddenDim):

                    norm = np.linalg.norm(self.gradients[key][:,:,i],"fro")
                    if norm > self.get_clip_mean(key)[i] + threshold* self.get_clip_stddev(key)[i]:
                        self.gradients[key][:,:,i] = self.get_clip_mean(key)[i]*(self.gradients[key][:,:,i]/norm)
                        print_flag = True
            if print_flag and print_if_clipped:
                print(f"Gradient {key} has been clipped! ---- Norm: {norm}, Mean: {self.mean[key]}, Stddev: {self.get_clip_stddev(key)}")


    def zero_gradient(self):
        """
        Zeros all gradients

        Returns:
        None
        """
        for key in self.gradients.keys():
            value = self.gradients[key]
            self.gradients[key] = np.zeros_like(value)


    def zero_grad_samples(self):
        """
        Zeros gradients for individual samples

        Returns:
        None
        """
        for key in self.gradients_sample.keys():
            value = self.gradients_sample[key]
            self.gradients_sample[key] = np.zeros_like(value)
        
    def gradient(self,ypred,yactual,xactual): 

        """
        Calculates the gradient with respect to its parameters.
        Used for batch data

        Parameters:
        - ypred (numpy array): The predicted values.
        - yactual (numpy array): The actual values.
        - xactual (numpy array): The input data.
        """

        if self.use_tensors:
            self.grad_tensor(ypred,yactual,xactual)
            return
        for sample_num in range(len(yactual)):
            self.grad_sample(ypred[sample_num],yactual[sample_num],sample_num,xactual[sample_num])
            self.updateSampleWeights()
        self.zero_grad_samples()


    def grad_sample(self,ypred_sample,yactual_sample,sample_no,xinput_sample):
        """
        Calculates the gradient for a specific sample.

        Parameters:
        - ypred_sample (numpy array): The predicted values for the specific sample.
        - yactual_sample (numpy array): The actual values for the specific sample.
        - sample_no (int): The index or identifier of the sample.
        - xinput_sample (numpy array): The input data for the specific sample.

        Returns:
        None
        """

        if not self.classifier:
            output_error  = self.costFunc(ypred_sample,yactual_sample,derivative=True)
            delta_prev = (np.ones((1,self.outputDim))*output_error) ## for arbitrary output layers multiply by grad z
        else:
            delta_prev = (ypred_sample-yactual_sample).reshape(self.outputDim,1).T
        self.gradients_sample["weights_out"] = self.a[:,-1,sample_no].reshape(self.NPL,1) @ delta_prev
        self.gradients_sample["output_bias"] = delta_prev

        if self.HiddenDim >= 1:
            
            delta = (self.parameters["weights_out"].T @  delta_prev.T)*self.activationFunc(self.z[:,self.HiddenDim-1,sample_no],derivative=True).transpose().reshape(self.NPL,1)
            self.gradients_sample["weights"][:,:,self.HiddenDim-1] = self.a[:,-2,sample_no].reshape(self.NPL,1).transpose() @ delta
            self.gradients_sample["bias"][:,-1] = delta.flatten()
            delta_prev = delta
        ## Hidden gradients
        if self.HiddenDim > 1:
            for i in range(self.HiddenDim -2, -1, -1):
                delta = (self.parameters["weights"][:,:,i].T @  delta_prev)*self.activationFunc(self.z[:,i,sample_no],derivative=True).transpose().reshape(self.NPL,1)
                self.gradients_sample["weights"][:,:,i] = self.a[:,i,sample_no].reshape(self.NPL,1).transpose() @ delta
                self.gradients_sample["bias"][:,i] = delta.flatten()
                delta_prev = delta
        # input layers
        
        if self.HiddenDim==0:
            delta = (self.parameters["weights_out"].T @  delta_prev).T
            self.gradients_sample["weights_in"] = xinput_sample.reshape((len(xinput_sample),1)) @ delta
            self.gradients_sample["input_bias"] = delta
        else:
            delta = (self.parameters["weights"][:,:,0].T @  delta_prev).T * self.activationFunc(self.z[:,0,sample_no],derivative=True)
            self.gradients_sample["weights_in"] = xinput_sample.reshape((len(xinput_sample),1)) @ delta
            self.gradients_sample["input_bias"] = delta

    def grad_tensor(self,ypred,yactual,xactual):
        """
        Calculates the gradient tensor with respect to the model parameters.

        Parameters:
        - ypred (numpy array or tensor): The predicted values.
        - yactual (numpy array or tensor): The actual values.
        - xactual (numpy array or tensor): The input data.

        Returns:
        None
        """
        if self.classifier:
            delta = (ypred-yactual)
        else:
            raise NotImplementedError
        self.gradients["weights_out"] = ((self.a[:,-1,:] @ delta).T)/self.minibatch_sz
        delta = delta.T
        self.gradients_sample["output_bias"] = delta/self.minibatch_sz
        if self.HiddenDim >= 1:
            for i in range(self.HiddenDim -1, -1, -1):
                if i==self.HiddenDim-1:
                    weights = self.parameters["weights_out"]
                else:
                    weights = self.parameters["weights"][:,:,i]

                delta = (weights.T @ delta)*self.activationFunc(self.z[:,i,:],derivative=True)
                self.gradients["weights"][:,:,i] = (self.a[:,i,:] @ delta.T)/self.minibatch_sz
                self.gradients["bias"][:,i] = np.mean(delta,axis=1)

            
            delta = (self.parameters["weights"][:,:,0].T @ delta) * self.activationFunc(self.z[:,0,:],derivative=True)
            self.gradients["weights_in"] = (delta @ xactual)/self.minibatch_sz
            self.gradients["input_bias"] = np.mean(delta,axis=1)
            
        else:
            raise NotImplementedError

        
    def updateSampleWeights(self, include_bias = False):
        """
        Updates the GRADIENTS for individual samples in the model.

        Parameters:
        - include_bias (bool): Whether to include bias terms in the weight update (default: False).

        Returns:
        None
        """
        for key in self.gradients_sample.keys():
            gradient_sample = self.gradients_sample[key]
            if key=="input_bias":
                gradient_sample = gradient_sample.T
            if (key=="bias" or key=="output_bias") and include_bias:
                continue 
            self.gradients[key] += gradient_sample/self.minibatch_sz


    def updateWeights(self,lr = .001,alg = "SGD",clip_gradients= False):
        """
        Updates the model weights using a specified optimization algorithm.

        Parameters:
        - lr (float): The learning rate for the optimization algorithm (default: 0.001).
        - alg (str): The optimization algorithm to use (default: "SGD").
        - clip_gradients (bool): Whether to clip gradients during the update (default: False).

        Returns:
        None
        """
        if clip_gradients:
            self.clip_gradients()
        if alg == "SGD":
            self.SGD(lr = lr)
        if alg == "ADAM":
            self.ADAM(lr = lr,beta1 = .75,beta2 = .999,epsilon=1e-8)
        if clip_gradients:
            self.update_clip_metrics()
        self.zero_gradient()
        


       
################################################################
#               OPTIMIZER FUNCTIONS
################################################################
    
    def SGD(self,lr=.0001):
        """
        Performs a Stochastic Gradient Descent (SGD) optimization step.

        Parameters:
        - lr (float): The learning rate for the SGD algorithm (default: 0.0001).

        Returns:
        None
        """
        for grad_name, grad_value in self.gradients.items():
            if grad_name=="output_bias": #or grad_name=="weights_out": #or grad_name=="weights_in":
                grad_value = grad_value.T
            if grad_name=="input_bias":
                grad_value = grad_value[:,np.newaxis]
            self.parameters[grad_name] -= lr* grad_value
            
    def ADAM(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Performs an ADAM optimization step.

        Parameters:
        - lr (float): The learning rate for the ADAM algorithm (default: 0.001).
        - beta1 (float): Exponential decay rate for the first moment estimates (default: 0.9).
        - beta2 (float): Exponential decay rate for the second moment estimates (default: 0.999).
        - epsilon (float): Small constant to prevent division by zero (default: 1e-8).

        Returns:
        None
        """
        for param_name, param_value in self.parameters.items():
            
            m = self.adam_m[param_name]
            v = self.adam_v[param_name]

            # Compute gradients 
            gradient = self.gradients.get(param_name, np.zeros_like(param_value))
            if param_name=="output_bias":
                gradient = gradient.T
            if param_name == "input_bias":
                gradient = gradient[:,np.newaxis]

            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1)
            v_hat = v / (1 - beta2)

            self.parameters[param_name] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            self.adam_m[param_name] = m
            self.adam_v[param_name] = v





##################################################################
                # LOSS FUNCTIONS
##################################################################
    @staticmethod
    def MSE(ypred,yactual,derivative = False):
        """
        Calculates the Mean Squared Error (MSE) or its derivative.

        Parameters:
        - ypred (numpy array): The predicted values.
        - yactual (numpy array): The actual values.
        - derivative (bool): If True, computes the derivative of MSE (default: False).

        Returns:
        numpy float or numpy array: If derivative is False, returns the MSE. If derivative is True,
        returns the derivative of MSE with respect to ypred.
        """
        if not derivative:
            return np.mean((ypred-yactual)**2)
        return 2*(ypred-yactual)/len(ypred)
    

    @staticmethod
    def cross_entropy_loss(y_pred, y_actual, epsilon=1e-15, derivative=False):
        """
        Calculates the Cross-Entropy Loss or its derivative.

        Parameters:
        - y_pred (numpy array): The predicted probabilities.
        - y_actual (numpy array): The actual binary labels (0 or 1).
        - epsilon (float): Small constant to prevent numerical instability (default: 1e-15).
        - derivative (bool): If True, computes the derivative of Cross-Entropy Loss (default: False).

        Returns:
        numpy float or numpy array: If derivative is False, returns the Cross-Entropy Loss.
        If derivative is True, returns the derivative of Cross-Entropy Loss with respect to y_pred.
        """

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_actual * np.log(y_pred)) / len(y_actual)
        if not derivative:
            return loss
        gradient = (y_pred - y_actual) / (y_pred * (1 - y_pred) * len(y_actual))
        return gradient


####################################################################
                # ACTIVATION FUNCTIONS
###################################################################
    @staticmethod
    def ReLU(x,derivative = False):
        if not derivative:
            return np.maximum(0, x)
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x, derivative=False):
        if not derivative:
            return np.tanh(x)
        return 1 - np.tanh(x)**2

    @staticmethod
    def sigmoid(x, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-x))
        return x * (1 - x)
    
    @staticmethod
    def silu(x, derivative=False):
        sigmoid_x = 1 / (1 + np.exp(-x))

        if not derivative:
            return x * sigmoid_x
        return sigmoid_x * (1 + x * (1 - sigmoid_x))
    @staticmethod
    def softmax(x, derivative=False):
        e_x = np.exp(x - np.max(x,axis=1)[:,np.newaxis]) 
        return e_x / e_x.sum(axis=1)[:,np.newaxis]
    @staticmethod
    def gelu(x, derivative=False):
        if not derivative:
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        else:
            ex = np.exp(-0.5 * x**2)
            c = np.sqrt(2 / np.pi)
            return 0.5 * (1 + np.tanh(c * (x + 0.044715 * x**3))) + 0.5 * x * c * ex * (1 / np.cosh(c * (x + 0.044715 * x**3)))**2 * (1 + 0.044715 * x**2 * (3 - c**2 * x * (x + 0.044715 * x**3)))

##########################################################
#                   MISC
##########################################################

    def update_clip_metrics(self,matrix_norm = "fro",vector_norm=2):
        """
        Updates metrics used for gradient clipping.

        Parameters:
        - matrix_norm (str): The norm to be used for matrix gradients (default: "fro").
        - vector_norm (int): The norm to be used for vector gradients (default: 2).

        Returns:
        None
        """
        self.n += 1
        for key in self.gradients.keys(): 
            if key != "bias" and key != "weights":
                norm = np.linalg.norm(self.gradients[key],vector_norm)
                delta = norm - self.mean[key]
                self.mean[key] += delta / self.n
                delta2 = norm - self.mean[key]
                self.M2[key] += delta * delta2
            
            elif key=="weights":
                for i in range(self.HiddenDim):
                    norm = np.linalg.norm(self.gradients[key][:,:,i],matrix_norm)
                    delta = norm - self.mean[key][i]
                    self.mean[key][i] += delta / self.n
                    delta2 = norm - self.mean[key][i]
                    self.M2[key][i] += delta * delta2
            else:
                for i in range(self.HiddenDim):
                    norm = np.linalg.norm(self.gradients[key][:,i],vector_norm)
                    delta = norm - self.mean[key][i]
                    self.mean[key][i] += delta / self.n
                    delta2 = norm - self.mean[key][i]
                    self.M2[key][i] += delta * delta2
            #print("saved_val: ",key,norm)

                    

    ## Helper function
    def get_clip_mean(self,key):
        return self.mean[key] if self.n > 0 else np.nan
    ## Helper function
    def get_clip_stddev(self,key):
        return np.sqrt(self.M2[key] / (self.n)) if self.n > 1 else np.nan

##########################################################################
#                   NON-class methods
########################################################################
    
@staticmethod
def save_state(object,file_name,directory="/checkpoints"):
    """
    Saves the state of an object to a file.

    Parameters:
    - object: The object whose state will be saved.
    - file_name (str): The name of the file to save the object state.
    - directory (str): The directory where the file will be saved (default: "/checkpoints").

    Returns:
    None
    """
    current_directory = os.getcwd()
    dir = current_directory+directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    pickle_file_path = "checkpoints/"+file_name+".pkl"
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(object, file)

@staticmethod      
def load_state(file_path):
    """
    Loads the state of an object from a file.

    Parameters:
    - file_path (str): The path to the file containing the object state.

    Returns:
    object: The object with the loaded state.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


@staticmethod
def validate_data(y_pred,yactual):
    """
    Computes the percentage of correct predictions.

    Parameters:
    - y_pred (numpy array): The predicted values.
    - y_actual (numpy array): The actual values.

    Returns:
    float: The percentage of correct predictions between y_pred and y_actual.
    """
    count = 0
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i])==np.argmax(yactual[i]):
            count+=1
    return 100*(count/len(y_pred))
    





            

           


















