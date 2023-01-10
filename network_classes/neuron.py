# for class Neuron


from math import exp


class Neuron:
    '''
    * Changing to forward propagating
    
    * Using TanH for Activation Function
        - https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
    
    '''
    def __init__(self, bias=0, receiving=[], sending=[]):
        # for calculating the value
        self.bias           = bias
        self.receiving      = []
        self.value          = 0
        
        # one value of connections list = (layer,index) of connection, weight 
        # * example: ( (1,1), 0.5 )
        self.sending        = []


    # activation functions
    # ---------------------

    # tanh activation function
    def tanh(self, x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    def tanh_calculate_value(self):
        # do tanh to verything
        value_sum = self.tanh(self.bias)
        for value in self.receiving:
            value_sum += self.tanh(value)
        # final value is the sum
        self.value = value_sum


    # original version
    def original_calculate_value(self):
        raw_value   = sum(self.receiving) + self.bias
        self.value  = self.tanh(raw_value)


    def my_activation_function(self):
        raw_value = sum(self.receiving) + self.bias
        if raw_value > 0:
            self.value = 1
        else: 
            self.value = 0



    # updates self.value
    # -------------------
    def calculate_value(self):
        # chosen activation function
        self.my_activation_function()

        # reset receiving list for next calculation
        self.receiving = []












