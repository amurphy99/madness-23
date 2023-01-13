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
        self.mut_connection_chance  = 0.60

        self.mut_neuron_chance      = 0.30



        # Layers
        # ---------------------------------------------------
        # ---------------------------------------------------
        # going to leave input layer as a list still
        # for others, needs to be dictionary
        # * using: new_neuron_id(self, layer_index)

        self.layers = []


        # input layer
        # ------------
        self.layers.append( {} )
        for j in range(num_inputs):
            self.add_new_neuron(0, add_connections=False)


        # middle layers
        # --------------
        num_neurons = 7 # num_inputs
        for i in range(self.starting_layers):
            # create empty dict for the new layerand fill empty dict with neurons
            self.layers.append( {} )
            # layer index of this layer will always be i+1 because the input layer is already in
            for j in range(num_neurons):
                self.add_new_neuron(i+1, add_connections=False)

        
        # final layer
        # ------------
        self.layers.append( {} )
        self.layers[-1]["final_key"] = Neuron(0)


        # connections
        # ------------
        # (all but the final two layers)
        for i in range( len(self.layers)-2 ):
            for neuron in self.layers[i].values():
                for j in range(self.starting_connections):
                    # new connections in ONLY the following layer, or all following layers besides the FINAL layer
                    self.new_connection(neuron, (i+1,i+1))                  # only the next
                    #self.new_connection(neuron, (i+1,len(self.layers)-1))  # all but final layer


        # add connections to the final neuron
        for neuron in self.layers[-2].values():
            neuron.sending.append( [(-1,"final_key"), 1.0] )




    # Calculates all of the neuron values
    # ------------------------------------
    def calculate_value(self):
        # starting with the first layer
        for layer in self.layers:
            for neuron in list(layer.values()):

                # first calculate the final value
                neuron.calculate_value()

                # next send that value to all of the connections
                for connection in neuron.sending:
                    address = connection[0]
                    value   = connection[1] * neuron.value
                    self.layers[ address[0] ][ address[1] ].receiving.append(value)

        # return final neurons value
        return self.layers[-1]["final_key"].value


    # set new inputs for a calculation
    def set_inputs(self, new_inputs):
        # quick error checking
        if len(new_inputs) != len(self.layers[0]):
            print("Error: wrong number of inputs; agent takes {} inputs but received {} values".format( len(self.layers[0]), len(new_inputs) ))

        # setting the inputs
        input_layer = list(self.layers[0].values())
        for i in range(len(new_inputs)):
            input_neuron = input_layer[i]
            input_value  = new_inputs[i]
            # broken up just in case i wanted to do somethign with the inputs first
            input_neuron.receiving.append(input_value)




    # Creation and Deletion of neurons
    # -------------------------------------------

    # Create a Neuron with only the bias defined
    def new_neuron(self):
        new_bias = random.uniform(-self.bias_range, self.bias_range)
        return Neuron(new_bias)


    # Generate new neuron address key
    def new_neuron_id(self, layer_index):
        # start with blank key so we can enter the loop
        as_str = ""
        while as_str == "" or as_str in list(self.layers[layer_index].keys()):
            # generate random key using an integer into a string
            as_int = random.randint(0,1000)
            as_str = str(as_int)

        # return the newly created key
        return as_str


    # Create new FULL neuron and add it to the layers (also adds initial connections to it)
    def add_new_neuron(self, layer_index, add_connections=True):
        # create base neuron and a key for it
        base_neuron = self.new_neuron()
        address_key = self.new_neuron_id(layer_index)
        self.layers[layer_index][address_key] = base_neuron

        # we only want to add connections sometimes, can specify in function call
        if add_connections:
            for j in range(self.starting_connections):
                # new connections in ONLY the following layer, or all following layers besides the FINAL layer
                self.new_connection(base_neuron, (layer_index+1,layer_index+1))     # only the next
                #self.new_connection(neuron, (layer_index+1,len(self.layers)-1))    # all but final layer




    # Deleting Neurons
    def delete_neuron(self, address):
        '''
        * address = (i, "key_string")
        * need to delete all connections and the neuron itself

        '''
        # delete all connections TO this neuron
        # --------------------------------------
        # look for all existing connections to this neuron and remove them
        #              first layer,      layer being deleted
        #print(address)
        for i in range(0, address[0]):
            for neuron in self.layers[i].values():
                #print(neuron.sending)
                # list to delete later
                to_pop = []
                for j in range(len(neuron.sending)):
                    # neuron.sending[j] -> (address, weight)
                    if neuron.sending[j][0][0] == address[0] and neuron.sending[j][0][1] == address[1]:
                        to_pop.append(j)

                # delete all marked connections
                # start from the highest index or else they will change after each one is popped
                to_pop.reverse()
                for index in to_pop:
                    neuron.sending.pop(index)
                #print()
                #print(neuron.sending)
                #print("\n")

        # delete the actual neuron
        # -------------------------
        # self.layers[address[0]]   -> correct dictionary from the layer list
        # address[1]                -> string key to the dictionary for this neuron
        self.layers[address[0]].pop(address[1])





    # Create / Delete Connections
    # ----------------------------

    # create connection
    def new_connection(self, neuron, layer_range):
        # pick a neuron to connect to
        selected_layer  = random.randint( layer_range[0], layer_range[1]                     )
        selected_index  = random.randint(              0, len(self.layers[selected_layer])-1 )
        selected_neuron = list(self.layers[selected_layer].keys())[selected_index] # need to turn the index # into the key
        address = [selected_layer, selected_neuron]

        # if already connected to this neuron, cancel this operation
        for existing_connection in neuron.sending:
            if address == existing_connection[0]: return

        # if not, finish creating new connection
        new_weight = random.uniform(-self.weight_range, self.weight_range)
        neuron.sending.append( [address, new_weight] )


    # delete connection
    def del_connection(self, neuron):
        # leave at least 1 connection
        if len(neuron.sending) > 2:
            selected_connection = random.randint(0, len(neuron.sending)-1)
            neuron.sending.pop( selected_connection )




    # Mutating functions
    # -------------------
    def mutate(self):
        '''
        * can only add/delete connections on layers before the final 2
            - final layer has nothing to connect to
            - second to last layer only connects to final layer

        '''
        # all layers besides final 2
        # ---------------------------
        for i in range( 1, len(self.layers)-1 ):

            # add/del neuron
            # -------------------
            if self.mut_neuron_chance > random.uniform(0, 1):
                if 0.5 > random.uniform(0, 1): # 50/50 to add or delete
                    self.add_new_neuron(i)
                else:
                    selected_index  = random.randint( 0, len(self.layers[i])-1 )
                    selected_neuron = list(self.layers[i].keys())[selected_index] # need to turn the index # into the key
                    address = [i, selected_neuron]
                    self.delete_neuron(address)



        # all layers besides final 2
        # ---------------------------
        for i in range( len(self.layers)-2 ):
            for neuron in self.layers[i].values(): # for each neuron in

                # add/del connection
                # -------------------
                if self.mut_connection_chance > random.uniform(0, 1):
                    if 0.5 > random.uniform(0, 1): # 50/50 to add or delete
                        self.new_connection(neuron, (i+1,i+1)) # self.new_connection(neuron, (i+1,len(self.layers)-2)) 
                    else:
                        self.del_connection(neuron)


        # all layers besides final
        # ---------------------------
        for i in range( len(self.layers)-1 ):
            # for each neuron in
            for neuron in self.layers[i].values():

                # mutate bias
                # ------------
                if self.bias_mutation_chance > random.uniform(0, 1):
                    neuron.bias += random.uniform(-self.bias_mutation_range, self.bias_mutation_range)

                    # limit the max/min?
                    if neuron.bias > 2: 
                        neuron.bias = 2.0

                    elif neuron.bias < -2: 
                        neuron.bias = -2.0

                # mutate connection weight
                # -------------------------
                for connection in neuron.sending: # for each connection in

                    if self.weight_mutation_chance > random.uniform(0, 1):
                        connection[1] += random.uniform(-self.weight_mutation_range, self.weight_mutation_range)
                        
                        # limit the max/min?
                        if connection[1] > 2: 
                            connection[1] = 2.0

                        elif connection[1] < -2: 
                            connection[1] = -2.0







    










    # Printing Visual
    # ----------------
    def print1(self):
        print("format  ->  final value    ") 
        print("            -------------- ")
        print("            (# to, # from)  \n\n")

        for i in range(len(self.layers)):
            line1 = "" # " " * 16
            line2 = "" # " " * 16
            for neuron in self.layers[i].values():
                line1 += "{:-5.2f} ".format(neuron.value)
                line2 += "({:-1},{:-1}) ".format( neuron.connections_to, len(neuron.sending) )
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

            for j in range(len(self.layers[i].values())):
                neuron = list(self.layers[i].values())[j]

                address = "({},{})".format(i, j)

                lines[0] += "{:6s} -> {:-5.2f}, {:-6.2f}   ".format(address, neuron.bias, neuron.value)
                lines[1] += ("-" * 23) + "   "

                for k in range(len(neuron.sending)):
                    conn = neuron.sending[k]

                    address = "({},{})".format(conn[0][0], conn[0][1])

                    lines[k+2] += "{:6s} -> {:-5.2f} | {:-5.2f}   ".format(address, conn[1], (conn[1]*neuron.value))

            for line in lines:
                print(line)
                    




























