
## TODO 
# Update this file to the current version
# will need this file for CUDA and C++ functions

import numpy as np
import matplotlib.pyplot as plt
cimport numpy as np




class NN(object):
    def __init__(self,name = "Neural Net"):
        self.name = name

class FullyConnected(NN):
    def __init__(self,inputDim,outputDim,NPL,HiddenDim,minibatch_sz, activation_fcn="ReLU",cost_func="MSE", optimizer="SGD"):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.NPL = NPL # Neurons per layer
        self.HiddenDim = HiddenDim
        self.minibatch_sz = minibatch_sz
        self.activation_fcn=activation_fcn
        self.optimizer = optimizer

        self.parameters = dict()
        self.gradients = dict()
        self.init_weights()
        self.init_gradients()
        self.init_intermediate_vals()
        self.init_activationFunc(activation_fcn)
        self.init_alg_params()
        
        self.costFunc = self.MSE
        self.mse_history = []

    def init_gradients(self):
        cdef float[:, :] gradients_weights_in = np.zeros((self.inputDim, self.NPL), dtype="float32")
        cdef float[:, :, :] gradients_weights = np.zeros((self.NPL, self.NPL, self.HiddenDim), dtype="float32")
        cdef float[:, :] gradients_weights_out = np.zeros((self.NPL, self.outputDim), dtype="float32")
        cdef float[:, :] gradients_bias = np.zeros((self.NPL, self.HiddenDim), dtype="float32")
        cdef float[:, :] gradients_output_bias = np.zeros((self.outputDim, 1), dtype="float32").T
        cdef float[:, :] gradients_input_bias = np.zeros((self.NPL, 1), dtype="float32")

        # Assign the gradients to the class attributes
        self.gradients = {
            "weights_in": gradients_weights_in,
            "weights": gradients_weights,
            "weights_out": gradients_weights_out,
            "bias": gradients_bias,
            "output_bias": gradients_output_bias,
            "input_bias": gradients_input_bias
        }

        # Do the same for gradients_sample
        self.gradients_sample = {
            "weights_in": np.zeros((self.inputDim, self.NPL), dtype="float32"),
            "weights": np.zeros((self.NPL, self.NPL, self.HiddenDim), dtype="float32"),
            "weights_out": np.zeros((self.NPL, self.outputDim), dtype="float32"),
            "bias": np.zeros((self.NPL, self.HiddenDim), dtype="float32"),
            "output_bias": np.zeros((self.outputDim, 1), dtype="float32").T,
            "input_bias": np.zeros((self.NPL, 1), dtype="float32")
        }


    def init_weights(self):
    
        cdef float[:, :] weights_in = ((np.random.randn(self.inputDim, self.NPL) * np.sqrt(1.0 / self.inputDim)).T).astype("float32")
        cdef float[:, :, :] weights = (np.random.randn(self.NPL, self.NPL, self.HiddenDim) * np.sqrt(2.0 / self.NPL)).astype("float32")
        cdef float[:, :] weights_out = ((np.random.randn(self.NPL, self.outputDim) * np.sqrt(1.0 / self.NPL)).T).astype("float32")
        cdef float[:, :] bias = np.zeros((self.NPL, self.HiddenDim), dtype="float32")
        cdef float[:, :] output_bias = np.zeros((self.outputDim, 1), dtype="float32")
        cdef float[:, :] input_bias = np.zeros((self.NPL, 1), dtype="float32")

        self.parameters = {
            "weights_in": weights_in,
            "weights": weights,
            "weights_out": weights_out,
            "bias": bias,
            "output_bias": output_bias,
            "input_bias": input_bias
        }


    def init_intermediate_vals(self):
        self.a = np.zeros((self.NPL, self.HiddenDim +1  ,self.minibatch_sz))
        self.z = np.zeros((self.NPL, self.HiddenDim +1,self.minibatch_sz))
    def init_alg_params(self):
        if self.optimizer == "ADAM":
             self.adam_m = {param_name: np.zeros_like(param_value) for param_name, param_value in self.parameters.items()}
             self.adam_v = {param_name: np.zeros_like(param_value) for param_name, param_value in self.parameters.items()}
        
            
    def init_activationFunc(self,activation_fcn):
        if activation_fcn == "ReLU":
            self.activationFunc = self.ReLU
        elif activation_fcn == "Tanh":
            self.activationFunc = self.tanh
        elif activation_fcn == "Sigmoid":
            self.activationFunc = self.sigmoid
        elif activation_fcn == "SiLU":
            self.activationFunc = self.silu
        else:
            raise RuntimeError("Unspecified activation function")
    
    def forward_sample(self,x_initial,sample_no): 
            x_init = x_initial.reshape(self.inputDim,1)
            self.z[:,0:1,sample_no] = self.parameters["weights_in"] @ x_init + self.parameters["input_bias"]
            self.a[:,0:1,sample_no] = self.parameters["weights_in"] @ x_init + self.parameters["input_bias"]

            for i in range(self.HiddenDim):
                self.z[:,i+1,sample_no] = self.parameters["weights"][:, :, i] @ self.a[:, i,sample_no] +self.parameters["bias"][:, i-1]
                self.a[:,i+1,sample_no] = self.activationFunc(self.z[:,i,sample_no],derivative=False)

            return self.parameters["weights_out"] @ self.a[:,self.HiddenDim,sample_no] +self.parameters["output_bias"]
    

    def forward(self,x_minibatch):
        
        y_out = np.zeros((self.minibatch_sz,self.outputDim))
        for i,x in enumerate(x_minibatch):
            y_out[i,:] = self.forward_sample(x,sample_no=i)
        return y_out
    


    def clip_gradients(self, threshold):
        for key in self.gradients.keys():
            self.gradients
        
    def zero_gradient(self):
        for key in self.gradients.keys():
            value = self.gradients[key]
            self.gradients[key] = np.zeros_like(value)
    def zero_grad_samples(self):
        for key in self.gradients_sample.keys():
            value = self.gradients_sample[key]
            self.gradients_sample[key] = np.zeros_like(value)
        
    def gradient(self,ypred,yactual,xactual):

        def grad_sample(self,ypred_sample,yactual_sample,sample_no,xinput_sample):
            output_error  = self.costFunc(ypred_sample,yactual_sample,derivative=True)
            delta_prev = (np.ones((1,self.outputDim))*output_error).T
            self.gradients_sample["weights_out"] = self.a[:,-1,sample_no].reshape(self.NPL,1) @ delta_prev
            self.gradients_sample["output_bias"] = delta_prev

            if self.HiddenDim >= 1:
                
                delta = (self.parameters["weights_out"].T @  delta_prev)*self.activationFunc(self.z[:,self.HiddenDim-1,sample_no],derivative=True).transpose().reshape(self.NPL,1)
                self.gradients_sample["weights"][:,:,self.HiddenDim-1] = self.a[:,-2,sample_no].reshape(self.NPL,1).transpose() @ delta
                self.gradients_sample["bias"][:,-1] = delta.reshape(-1)
                delta_prev = delta
            ## Hidden gradients
            if self.HiddenDim > 1:
                for i in range(self.HiddenDim -2, 0, -1):
                    delta = (self.parameters["weights"][:,:,i].T @  delta_prev)*self.activationFunc(self.z[:,i,sample_no],derivative=True).transpose().reshape(self.NPL,1)
                    self.gradients_sample["weights"][:,:,i] = self.a[:,i,sample_no].reshape(self.NPL,1).transpose() @ delta
                    self.grad_bias_sample[:,i] = delta.reshape(-1)
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

            

            
        for sample_num in range(len(yactual)):
            grad_sample(self,ypred[sample_num],yactual[sample_num],sample_num,xactual[sample_num])
            self.updateSampleWeights()
        
        self.zero_grad_samples()
    def updateSampleWeights(self):
        for key in self.gradients_sample.keys():
            gradient_sample = self.gradients_sample[key]
            if key=="output_bias" or key=="input_bias":
                gradient_sample = gradient_sample.T
            self.gradients[key] += gradient_sample/self.minibatch_sz


        
        
        

    def updateWeights(self,lr = .001,alg = "SGD",beta1 = .9,beta2 = .999,epsilon=1e-8):
        if alg == "SGD":
            self.SGD(lr = lr)
        if alg == "ADAM":
            self.ADAM(lr = .001,beta1 = .9,beta2 = .999,epsilon=1e-8)
        self.zero_gradient()


       
################################################################
#               OPTIMIZER FUNCTIONS
################################################################
    
    def SGD(self,lr=.0001):
        for grad_name, grad_value in self.gradients.items():
            if grad_name == "weights_out" or grad_name == "weights_in":
                grad_value = grad_value.T
            self.parameters[grad_name] -= lr* grad_value
    def ADAM(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for param_name, param_value in self.parameters.items():
            # Retrieve ADAM variables
            m = self.adam_m[param_name]
            v = self.adam_v[param_name]

            # Compute gradients (assuming you have a gradients dictionary)
            gradient = self.gradients.get(param_name, np.zeros_like(param_value))
            if param_name=="weights_out" or param_name=="weights_in":
                gradient = gradient.T

            # Update moments
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # Bias-corrected moments
            m_hat = m / (1 - beta1)
            v_hat = v / (1 - beta2)

            # Update parameters
            self.parameters[param_name] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

            # Update ADAM variables
            self.adam_m[param_name] = m
            self.adam_v[param_name] = v





##################################################################
                # LOSS FUNCTIONS
##################################################################
    @staticmethod
    def MSE(ypred,yactual,derivative = False):
        if not derivative:
            return np.mean((ypred-yactual)**2)
        return 2*(ypred-yactual)/len(ypred)
    @staticmethod
    def cross_entropy_loss(y_pred, y_actual, epsilon=1e-15, derivative=False):

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Calculate cross-entropy loss
        loss = -np.sum(y_actual * np.log(y_pred)) / len(y_actual)

        if not derivative:
            return loss

        # Calculate the gradient of the cross-entropy loss
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






        

           


















