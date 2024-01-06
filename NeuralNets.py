import numpy as np
import pickle
import os

class NN(object):
    def __init__(self,name = "Neural Net"):
        self.name = name

class FullyConnected(NN):
    def __init__(self,inputDim,outputDim,NPL,HiddenDim,minibatch_sz, activation_fcn="SiLU",cost_func="MSE", optimizer="SGD",classifier=False,dtype_ = np.float32, use_tensors=True,cost_function = "CrossEntropyLoss"):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.NPL = NPL # Neurons per layer
        self.HiddenDim = HiddenDim
        self.minibatch_sz = minibatch_sz
        self.activation_fcn=activation_fcn
        self.optimizer = optimizer
        self.classifier = classifier
        self.dtype_ = dtype_
        self.use_tensors = use_tensors

        self.parameters = dict()
        self.gradients = dict()
        self.init_weights()
        self.init_gradients()
        self.init_intermediate_vals()
        self.init_activationFunc(activation_fcn)
        self.init_alg_params()
        self.init_gradient_clipping()
        
        self.init_costFunc(cost_func)
        self.loss_history = []
        self.train_validation =[]
        self.test_validation = []

    def init_gradients(self):
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
        self.parameters = {
            "weights_in": ((np.random.randn(self.inputDim, self.NPL) * np.sqrt(1.0 / self.inputDim)).T).astype(dtype=self.dtype_),
            "weights": np.random.randn(self.NPL, self.NPL, self.HiddenDim) * np.sqrt(2.0 / self.NPL),
            "weights_out": (np.random.randn(self.NPL, self.outputDim) * np.sqrt(1.0 / self.NPL)).T,
            "bias": np.zeros((self.NPL, self.HiddenDim)),
            "output_bias": np.zeros((self.outputDim, 1)),
            "input_bias": np.zeros((self.NPL, 1))
        }

    def init_intermediate_vals(self):
        self.a = np.zeros((self.NPL, self.HiddenDim +1  ,self.minibatch_sz))
        self.z = np.zeros((self.NPL, self.HiddenDim +1,self.minibatch_sz))
        if self.classifier is True:
            self.z_out = np.zeros((self.outputDim,self.minibatch_sz))
            
        
    def init_alg_params(self):
        if self.optimizer == "ADAM":
             self.adam_m = {param_name: np.zeros_like(param_value) for param_name, param_value in self.parameters.items()}
             self.adam_v = {param_name: np.zeros_like(param_value) for param_name, param_value in self.parameters.items()}
        
    def init_gradient_clipping(self):
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
        if loss_function=="CrossEntropyLoss":
            self.costFunc = self.cross_entropy_loss

    def forward_sample(self,x_initial,sample_no): 
        x_initial = x_initial.reshape(self.inputDim,1)
        
        self.z[:,0:1,sample_no] = self.parameters["weights_in"] @ x_initial + self.parameters["input_bias"]
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
        if self.use_tensors:
            return self.forward_tensors(x_minibatch)
        y_out = np.zeros((self.minibatch_sz,self.outputDim))
        for i,x in enumerate(x_minibatch):
            y_out[i,:] = self.forward_sample(x,sample_no=i).flatten()
        return y_out
    


    def forward_tensors(self,x_minibatch):
        self.z[:,0:1,:] = np.tensordot(self.parameters["weights_in"][:,np.newaxis,:], x_minibatch.T ,axes=1) + self.parameters["input_bias"][:,np.newaxis]
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
        ## Uses gradient norm within 3 standard deviations
        ## here, threshold represents the number of standard devations from the mean

        ## needed to gather samples before we clip
        
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
                    #print(key,norm)
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
        for key in self.gradients.keys():
            value = self.gradients[key]
            self.gradients[key] = np.zeros_like(value)
    def zero_grad_samples(self):
        for key in self.gradients_sample.keys():
            value = self.gradients_sample[key]
            self.gradients_sample[key] = np.zeros_like(value)
        
    def gradient(self,ypred,yactual,xactual): 
        if self.use_tensors:
            self.grad_tensor(ypred,yactual,xactual)
            return
        for sample_num in range(len(yactual)):
            self.grad_sample(ypred[sample_num],yactual[sample_num],sample_num,xactual[sample_num])
            self.updateSampleWeights()
        self.zero_grad_samples()

    def grad_sample(self,ypred_sample,yactual_sample,sample_no,xinput_sample):
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

            delta = (self.parameters["weights"][:,:,0].T @ delta ) * self.activationFunc(self.z[:,0,:],derivative=True)
            self.gradients["weights_in"] = (delta @ xactual)/self.minibatch_sz
            self.gradients["input_bias"] = np.mean(delta,axis=1)
            
        else:
            raise NotImplementedError


        
    



           
        
    def updateSampleWeights(self, include_bias = False):
        for key in self.gradients_sample.keys():
            gradient_sample = self.gradients_sample[key]
            if key=="input_bias":
                gradient_sample = gradient_sample.T
            if (key=="bias" or key=="output_bias") and include_bias:
                continue 
            self.gradients[key] += gradient_sample/self.minibatch_sz


        
        
        

    def updateWeights(self,lr = .001,alg = "SGD",clip_gradients= False):
        if clip_gradients:
            self.clip_gradients()
        if alg == "SGD":
            self.SGD(lr = lr)
        if alg == "ADAM":
            self.ADAM(lr = .001,beta1 = .9,beta2 = .999,epsilon=1e-8)
        if clip_gradients:
            self.update_clip_metrics()
        self.zero_gradient()
        


       
################################################################
#               OPTIMIZER FUNCTIONS
################################################################
    
    def SGD(self,lr=.0001):
        for grad_name, grad_value in self.gradients.items():
            if grad_name=="output_bias": #or grad_name=="weights_out": #or grad_name=="weights_in":
                grad_value = grad_value.T
            if grad_name=="input_bias":
                grad_value = grad_value[:,np.newaxis]
            #print(self.parameters[grad_name].shape,grad_name,grad_value.shape)
            self.parameters[grad_name] -= lr* grad_value
            
    def ADAM(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for param_name, param_value in self.parameters.items():
            # Retrieve ADAM variables
            m = self.adam_m[param_name]
            v = self.adam_v[param_name]

            # Compute gradients (assuming you have a gradients dictionary)
            gradient = self.gradients.get(param_name, np.zeros_like(param_value))
            if param_name=="output_bias":
                gradient = gradient.T
            if param_name == "input_bias":
                gradient = gradient[:,np.newaxis]

            # Update moments
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # Bias-corrected moments
            m_hat = m / (1 - beta1)
            v_hat = v / (1 - beta2)

            # Update parameters
            #print(param_name)
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

## Gradient clipping functions
    def update_clip_metrics(self,matrix_norm = "fro",vector_norm=2):
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

                    


    def get_clip_mean(self,key):
        return self.mean[key] if self.n > 0 else np.nan

    def get_clip_stddev(self,key):
        return np.sqrt(self.M2[key] / (self.n)) if self.n > 1 else np.nan

##########################################################################
#                   NON-class methods
########################################################################
@staticmethod
def save_state(object,file_name,directory="/checkpoints"):
    current_directory = os.getcwd()
    dir = current_directory+directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    pickle_file_path = "checkpoints/"+file_name+".pkl"
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(object, file)
def load_state(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
# Returns percent error 
@staticmethod
def validate_data(y_pred,yactual):
    count = 0
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i])==np.argmax(yactual[i]):
            count+=1
    return 100*(count/len(y_pred))
    





            

           


















