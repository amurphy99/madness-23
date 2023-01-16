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

        # statistics
        self.connections_to = 0







    # activation functions
    # ---------------------



    # tanh activation function
    # -------------------------
    def tanh(self, x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    def tanh_after_sum(self):
        raw_value  = sum(self.receiving) + self.bias
        self.value = self.tanh(raw_value)

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
    def my_activation_function(self):
        raw_value = sum(self.receiving) + self.bias
        if raw_value > 0:
            self.value = 1.0
        else: 
            self.value = -1.0





    # updates self.value
    # -------------------
    def calculate_value(self):
        # chosen activation function
        self.tanh_after_sum()

        # reset receiving list for next calculation
        self.connections_to = len(self.receiving)
        self.receiving = []











