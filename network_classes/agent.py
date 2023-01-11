# for class Agent


import random

from .neuron import Neuron


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


    given number of inputs, given species

    species:
        * number of layers
        * number of starting connection
        * connection starting weights
        * connection mutation weights
        * bias will just us ethe same shit

    for now we will have 1 starting layer in the middle

    species
        * number of layers = 2
        * number of starting connections = 4
        * range for starting weights and bias = (-1,1)

        * odds to add or delete a connection


    '''
    def __init__(self, num_inputs):

        # Species Creation Info
        # ----------------------
        self.starting_layers      = 3
        self.starting_connections = 4

        self.bias_range   = 1.0
        self.weight_range = 1.0


        # Species Mutation Info
        # ----------------------
        self.bias_mutation_range   = self.bias_range   / 2.0
        self.weight_mutation_range = self.weight_range / 2.0

        self.bias_mutation_chance   = 0.80
        self.weight_mutation_chance = 0.80
        self.mut_connection_chance  = 0.30



        # Layers
        # -------
        self.layers = []


        # first do the input layer
        input_layer = []
        for i in range(num_inputs):
            input_layer.append( self.new_neuron() )
        self.layers.append(input_layer)


        # middle layers
        for i in range(self.starting_layers):
            new_layer = []
            for j in range(num_inputs):
                new_layer.append( self.new_neuron() )
            self.layers.append(new_layer)


        # final layer
        final_layer = [ Neuron(0) ]
        self.layers.append(final_layer)


        # add connections for all but the final two layers
        for i in range( len(self.layers)-2 ):
            for neuron in self.layers[i]:
                for j in range(self.starting_connections):
                    self.new_connection(neuron, (i+1,i+1))
                    #self.new_connection(neuron, (i+1,len(self.layers)-1))

        # add connections to the final neuron
        for neuron in self.layers[-2]:
            neuron.sending.append( ((-1,0), 1.0) )




    # Calculates all of the neuron values
    # ------------------------------------
    def calculate_value(self):
        # starting with the first layer
        for layer in self.layers:
            for neuron in layer:

                # first calculate the final value
                neuron.calculate_value()

                # next send that value to all of the connections
                for connection in neuron.sending:
                    address = connection[0]
                    value   = connection[1] * neuron.value
                    self.layers[ address[0] ][ address[1] ].receiving.append(value)

        # return final neurons value
        return self.layers[-1][0].value


    # set new inputs for a calculation
    def set_inputs(self, new_inputs):
        # quick error checking
        if len(new_inputs) != len(self.layers[0]):
            print("Error: wrong number of inputs; agent takes {} inputs but received {} values".format( len(self.layers[0]), len(new_inputs) ))

        # setting the inputs
        for i in range(len(new_inputs)):
            input_neuron = self.layers[0][i]
            input_value  = new_inputs[i]
            # broken up just in case i wanted to do somethign with the inputs first
            input_neuron.receiving.append(input_value)





    # Create a Neuron with only the bias defined
    # -------------------------------------------
    def new_neuron(self):
        new_bias = random.uniform(-self.bias_range, self.bias_range)
        return Neuron(new_bias)



    # Create a new connection or delete an existing connection 
    # ---------------------------------------------------------
    def new_connection(self, neuron, layer_range):
        '''
        neuron 
            * the neuron being worked on

        weight_range 
            * range of values the initial weight could be

        layer_range = (start_int, end_int)
            * indicis for the layers this neuron can make a connection to
            (if this is in the 2nd layer, it can connect the maybe just the 3rd layer, or from the 3rd layer to the final layer)
        '''
        # pick a neuron to connect to
        selected_layer  = random.randint( layer_range[0], layer_range[1]                     )
        selected_neuron = random.randint(              0, len(self.layers[selected_layer])-1 )
        address = (selected_layer, selected_neuron)

        # if already connected to this neuron, cancel this operation
        for existing_connection in neuron.sending:
            if address == existing_connection[0]: return

        # if not, finish creating new connection
        new_weight = random.uniform(-self.weight_range, self.weight_range)
        neuron.sending.append( [address, new_weight] )


    def del_connection(self, neuron):
        # leave at least 1 connection
        if len(neuron.sending) > 2:
            selected_connection = random.randint(0, len(neuron.sending)-1)
            neuron.sending.pop( selected_connection )




    # Mutating functions
    # -------------------
    def mutate(self):
        # all layers besides final 2
        # ---------------------------
        for i in range( len(self.layers)-2 ):

            # for each neuron in
            for neuron in self.layers[i]:

                # mutate bias
                # ------------
                if self.bias_mutation_chance > random.uniform(0, 1):
                    neuron.bias += random.uniform(-self.bias_mutation_range, self.bias_mutation_range)

                    # limit the max/min?
                    #if neuron.bias > 1: 
                        #neuron.bias = 1.0

                    #elif neuron.bias < -1: 
                        #neuron.bias = -1.0


                # add/del connection
                # -------------------
                if self.mut_connection_chance > random.uniform(0, 1):
                    # 50/50 to add or delete
                    if 0.5 > random.uniform(0, 1):
                        #self.new_connection(neuron, (i+1,i+1))
                        self.new_connection(neuron, (i+1,len(self.layers)-1))
                    else:
                        self.del_connection(neuron)

                # for each connection in
                for connection in neuron.sending:

                    # mutate connection weight
                    # -------------------------
                    if self.weight_mutation_chance > random.uniform(0, 1):
                        connection[1] += random.uniform(-self.weight_mutation_range, self.weight_mutation_range)
                        
                        # limit the max/min?
                        #if connection[1] > 1: 
                            #connection[1] = 1.0

                        #elif connection[1] < -1: 
                            #connection[1] = -1.0


        # second to last layer just mutate the bias
        for neuron in self.layers[-2]:

            # mutate bias
            # ------------
            if self.bias_mutation_chance > random.uniform(0, 1):
                neuron.bias += random.uniform(-self.bias_mutation_range, self.bias_mutation_range)

                # limit the max/min?
                #if neuron.bias > 1: 
                    #neuron.bias = 1.0

                #elif neuron.bias < -1: 
                    #neuron.bias = -1.0



    # Printing Visual
    # ----------------
    def print1(self):
        print("format  ->  final value   ") 
        print("            ------------- ")
        print("            # connections \n\n")

        for i in range(len(self.layers)):
            line1 = "" # " " * 16
            line2 = "" # " " * 16
            for neuron in self.layers[i]:
                line1 += "{:-5.2f} ".format(neuron.value)
                line2 += "{:-5} ".format( len(neuron.sending) )
            print(line1)
            print(line2)
            print()


    def print2(self):
        
        print("""
        (0,1) -> bias, final value
        ----------------------------
        (1,0) -> weight | value sent
        (1,2) -> weight | value sent

        \n\n""")


        for i in range(len(self.layers)):
            #         0   1   2   3    4
            lines = ["", "", "", "", "\n"]

            for j in range(len(self.layers[i])):
                neuron = self.layers[i][j]

                address = "({},{})".format(i, j)

                lines[0] += "{:6s} -> {:-5.2f}, {:-6.2f}   ".format(address, neuron.bias, neuron.value)
                lines[1] += ("-" * 23) + "   "

                for k in range(len(neuron.sending)):
                    conn = neuron.sending[k]

                    address = "({},{})".format(conn[0][0], conn[0][1])

                    lines[k+2] += "{:6s} -> {:-5.2f} | {:-5.2f}   ".format(address, conn[1], (conn[1]*neuron.value))

            for line in lines:
                print(line)
                    




























