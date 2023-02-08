# for class Neuron


from math import exp
from sys  import getsizeof
from copy import deepcopy
from time import time

import numpy as np


class Neuron:
    '''
    * Changing to forward propagating
    
    * Using TanH for Activation Function
        - https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
    
    '''
    def __init__(self, bias=0, weights=[], connections_to=1):
        # for calculating the value
        self.bias           = bias
        self.receiving      = 0     #self.receiving      = []
        self.value          = 0
        
        # one value of connections list = (layer,index) of connection, weight
        self.sending = [] # list of other Neurons #self.sending = [] # * example: ( (1,1), 0.5 )
        self.weights = deepcopy(weights)

        # statistics
        self.this_connections_to = 0
        self.connections_to      = connections_to



    def __sizeof__(self):
        total_size  = 0
        total_size += getsizeof(self.bias)
        total_size += getsizeof(self.receiving)
        total_size += getsizeof(self.value)

        total_size += getsizeof(self.this_connections_to)
        total_size += getsizeof(self.connections_to)
        
        total_size += getsizeof(self.sending)
        total_size += getsizeof(self.weights)
        
        for i in range(len(self.weights)):
            total_size += getsizeof(self.weights[i])

        if len(self.sending) != len(self.weights): print("unequal weights and connections")

        return total_size



    # custom "deepcopy" method
    # -------------------------
    def neuron_copy(self):
        return Neuron(bias=self.bias, weights=self.weights, connections_to=self.connections_to)



    # tanh activation function
    # -------------------------
    def tanh(self, x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    def tanh_after_sum(self):
        #raw_value  = sum(self.receiving) + self.bias
        raw_value = self.receiving + self.bias

        # on rare occasions inputs can exceed the tanh limit
        if      raw_value >  700: raw_value =  700
        elif    raw_value < -700: raw_value = -700

        self.value = self.tanh(raw_value)



    # calculating value
    # ------------------

    def send_value(self):
        send_time = time()

        #weighted_values = np.multiply(np.fromiter(self.weights, dtype=np.double), self.value) 
        for i in range(len(self.sending)):
            #self.sending[i].receiving += weighted_values[i]
            self.sending[i].receiving += (self.weights[i]*self.value)
            self.sending[i].this_connections_to += 1

        return 0,(time()-send_time)


    def calculate_value(self):
        # chosen activation function
        calc_time = time()
        self.tanh_after_sum()
        calc = (time()-calc_time)

        calc1, send = self.send_value()

        # reset receiving list for next calculation
        self.connections_to      = self.this_connections_to #self.connections_to = len(self.receiving)
        self.this_connections_to = 0
        self.receiving           = 0                        #self.receiving = []

        return calc, send
        








'''
    def tanh_before_sum(self):
        after_tanh = [ self.tanh(self.bias) ]
        for value in self.receiving:
            after_tanh.append( self.tanh(value) )
        self.value = sum(after_tanh)

    def tanh_before_after_sum(self):
        after_tanh = [ self.tanh(self.bias) ]
        for value in self.receiving:
            after_tanh.append( self.tanh(value) )
        self.value = self.tanh( sum(after_tanh) )


    # sigmoid activation function
    # ----------------------------
    def sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))

    def sigmoid_calculate_value(self):
        raw_value  = sum(self.receiving) + self.bias
        self.value = self.sigmoid(raw_value)


    # my function
    # ------------
    def my_activation_function(self):
        raw_value = sum(self.receiving) + self.bias
        if raw_value > 0:
            self.value = 1.0
        else: 
            self.value = -1.0
'''




