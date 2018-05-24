import numpy as np
from collections import Counter
from math import sqrt
import perceptron_errors



class Perceptron:
    """
    Description:
    ---
    Perceptron with a  sigmoid activation function for binary classification
    """
    
    def __init__(self, n_tuple, weights=None):
        """
        Parameters
        ---
        n_tuples : int
        x = (x1,x2...x_n)
        what is the dimension of the vector space of these vectors

        weights : np.array
        an array of weights. if None weights will be calculated randomly 
        about a normal
        distribution centered at zero

        References
        ---
        1. https://intoli.com/blog/neural-network-initialization/
        
        """
        if(n_tuple<=0):
            raise perceptron_errors.VectorError()
        else:
            self.n_tuple = n_tuple
            
        if weights==None:
            # n_tuples + 1 because bias needs a weight as well
            """
            weight will be calculated using the np.random.normal() method
            with standard deviation calculated based on [1]
            STANDARD DEVIATION(std) equal to 'variances which are 
            inversely proportional to the number of inputs 
            into each neuron.
            """
            std = sqrt(2.0/n_tuple+1)
            self.weights = np.random.normal(loc=0,scale=std,size=n_tuple+1)
           
        self.learning_rate = 0.05
        self.bias = 1
    
    @staticmethod
    def sigmoidal_activation(x):
        """
        Description
        ---
        Sigmoidal activation function. exists between 0 and 1. 
        will take in as input the weighted sum of x_n, weights, 
        and bias and output a integer value [0,1]

        Parameter
        ---
        x: float
        the weighted sum x_n variable and the weights and bias

        Returns
        ---
        0 or 1 (activation function where the threshold is 0.5)
        
        """
        result = 1.0 / (1 + np.power(np.e, -x))
        return 0 if result < 0.5 else 1
    
    def __call__(self, x_n):
        """
        Parameters
        ---
        x_n : list
        a python list representing the n-tuple real valued x variable
        each numerical value in the list should be float

        Returns
        ---
        Perceptron.sigmoid_function : int 
        will return either a zero or one
        """
        weighted_input = self.weights[:-1] * x_n 
        weighted_sum = weighted_input.sum() + self.bias *self.weights[-1]
        return Perceptron.sigmoidal_activation(weighted_sum)

    
    def adjust(self, target_result, calculated_result, in_data):
        error = target_result - calculated_result
        for i in range(len(in_data)):
            correction = error * in_data[i]  *self.learning_rate
            #print("weights: ", self.weights)
            #print(target_result, calculated_result, in_data, error, correction)
            self.weights[i] += correction 
        # correct the bias:
        correction = error * self.bias * self.learning_rate
        self.weights[-1] += correction 
     

