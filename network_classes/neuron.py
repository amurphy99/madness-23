# for class Neuron




class Neuron:
    '''
    * Changing to forward propagating
    
    * Using TanH for Activation Function


    neuron.bias

    neuron.value

    neuron.final_value

    neuron.connections
        one value of connections list = (layer,index) of connection, weight 
        example: ( (1,1), 0.5 )
    
    '''
    def __init__(self, bias, connections=[]):
        self.bias = bias
        self.connections = connections

        self.value = 0
        self.final_value = 0
    


    def resolve(self):
        self.value += self.bias
        if self.value < 0: 	self.final_value = -1
        else:				self.final_value =  1


        




