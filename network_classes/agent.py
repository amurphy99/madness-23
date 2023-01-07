# for class Agent


import random


from neuron import Neuron


class Agent:
    '''

    species:
        number of layers

        starting range for weight and bias
        mutate by 0.1

    * layers are in reverse order
        - final layer is 0


        starting connection chance = 1.0
        random.uniform(0,1)



    '''
    def __init__(self, species, num_inputs):
        self.species    = species
        
        self.layers     = []

        # final neuron
        # -------------
        # - essentially a regular neuron in its own final layer
        # - bias of 0, no connections
        self.layers.append([ Neuron(bias=0, connections=[]) ])


        # first layer will have the same number of neurons as the inputs
        new_layer = []
        for i in range(num_inputs):
            # bias is random from a range given by species
            # start with connection
            # need bias
            new_bias    = random.uniform( -species.bias_range,      species.bias_range      )

            # first layers only connection is to the final layer
            new_weight  = random.uniform( -species.weight_range,    species.weight_range    )
            new_connections = [ ( (0,0), new_weight ) ]









    def resolve(self):

        for layer in self.layers:
            for neuron in layer:
                neuron.resolve()

                for conn in neuron.connections:
                    # conn[0] = tuple of target neuron's indicis
                    # conn[1] = connection weight
                    self.layers[ conn[0][0] ][ conn[0][1] ].value += (neuron.final_value * conn[1])














