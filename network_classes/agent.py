# for class Agent


import random
import time

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
        self.starting_layers      =  3
        self.starting_neurons     =  num_inputs 
        self.starting_connections =  4

        self.bias_range   = 1.0
        self.weight_range = 1.0

        self.max_bias   = 1.5
        self.max_weight = 1.5

        # Species Mutation Info
        # ----------------------
        self.bias_mutation_range   = self.bias_range   / 2.0
        self.weight_mutation_range = self.weight_range / 2.0

        self.bias_mutation_chance   = 0.80
        self.weight_mutation_chance = 0.80
        self.mut_connection_chance  = 0.60

        self.mut_neuron_chance      = 0.30


        # stats
        # ------
        self.time_deleting = 0



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
        for i in range(self.starting_layers):
            # create empty dict for the new layerand fill empty dict with neurons
            self.layers.append( {} )
            # layer index of this layer will always be i+1 because the input layer is already in
            for j in range(self.starting_neurons):
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
        for i in range(len(self.layers)):
            layer = self.layers[i]
            for neuron in list(layer.values()):

                # first calculate the final value
                if i != 0:
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

            input_neuron.value = input_value # for an experiment with non-tanh output from input neurons



    # Creation and Deletion of neurons
    # -------------------------------------------

    # Create a Neuron with only the bias defined
    def new_neuron(self):
        #new_bias = random.uniform(-self.bias_range, self.bias_range)
        new_bias = 0
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
        start_time = time.time()
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


        # delete the actual neuron
        # -------------------------
        # self.layers[address[0]]   -> correct dictionary from the layer list
        # address[1]                -> string key to the dictionary for this neuron
        self.layers[address[0]].pop(address[1])
        self.time_deleting += (time.time()-start_time)





    # Create / Delete Connections
    # ----------------------------

    # create connection
    def new_connection(self, neuron, layer_range):
        # get a list of available connection choices
        available_addresses = []
        for i in range(layer_range[0],layer_range[1]+1):
            for key in self.layers[i].keys(): 
                address = [i,key] # for each possible address
                not_in  = True    # check if it is already in this neurons sending list
                for existing_connection in neuron.sending:
                    if address == existing_connection[0]: not_in = False
                # if not in the sending list, add to possible connections
                if not_in: available_addresses.append(address)

        # if there are no new possible connections, do nothing
        # otherwise, pick an address from the list at random to connect to
        if   len(available_addresses) == 0: return
        elif len(available_addresses) == 1: selected_address_index = 0 # i dont think randint works with 0,0 range
        else:                               selected_address_index = random.randint(0, len(available_addresses)-1)

        # finish creating new connection
        address = available_addresses[selected_address_index]
        new_weight = random.uniform(-self.weight_range, self.weight_range)
        neuron.sending.append( [address, new_weight] )


    # (original new connection function)
    # * gives up if randomly picked connection already exists
    def new_connection_v0(self, neuron, layer_range):
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
        if len(neuron.sending) > 1:
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

            # delete any neurons with 0 connections to
            # -----------------------------------------
            addresses_to_delete = []
            for key in self.layers[i].keys():
                if self.layers[i][key].connections_to == 0:
                    addresses_to_delete.append( [i,key] )

            # delete addresses in a different loop so dictionary doesn't change size during iteration
            for address in addresses_to_delete:
                if len(self.layers[i]) > 4:
                    self.delete_neuron( address )


            # add/del neuron
            # -------------------
            if self.mut_neuron_chance > random.uniform(0, 1):
                if 0.5 > random.uniform(0, 1): # 50/50 to add or delete
                    if len(self.layers[i]) < (1.25*len(self.layers[0])):
                        self.add_new_neuron(i)
                else:
                    # * skip a deletion if one was already deleted
                    if len(self.layers[i]) > 4 and len(addresses_to_delete)==0:
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
                        self.new_connection(neuron, (i+1,i+1)) 
                        #self.new_connection(neuron, (i+1,len(self.layers)-2)) 
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
                    if neuron.bias > self.max_bias: 
                        neuron.bias = self.max_bias

                    elif neuron.bias < -self.max_bias: 
                        neuron.bias = -self.max_bias

                # mutate connection weight
                # -------------------------
                for connection in neuron.sending: # for each connection in

                    if self.weight_mutation_chance > random.uniform(0, 1):
                        connection[1] += random.uniform(-self.weight_mutation_range, self.weight_mutation_range)
                        
                        # limit the max/min?
                        if connection[1] > self.max_weight: 
                            connection[1] = self.max_weight

                        elif connection[1] < -self.max_weight: 
                            connection[1] = -self.max_weight







    












    # Printing Visual
    # ----------------

    def print1(self):
        print("format:")
        print("")
        print(" # neurons    |  final value        ") 
        print(" ------------ |  --------------     ")
        print(" average from |  (# to, # from) \n\n")

        print("          FGM    FGA   FGM3   FGA3    FTM    FTA   RDif    Ast     TO    Stl    Blk     PF")
             #   12 |  -3.25   4.71  -2.62  -1.00   5.79   8.50  -1.38  -2.62   0.29   0.08   0.58   3.67 
             #  4.1 |  (0,4)  (0,4)  (0,5)  (0,3)  (0,4)  (0,4)  (0,5)  (0,5)  (0,4)  (0,3)  (0,4)  (0,4)


        total_neurons     = 0
        total_connections = 0

        for i in range(len(self.layers)):
            # prep for the printed lines
            line1 = "" 
            line2 = ""

            # prep for data
            total_connections_to   = 0
            total_connections_from = 0
            for neuron in self.layers[i].values():
                # tracking data for later
                total_connections_to   += neuron.connections_to
                total_connections_from += len(neuron.sending)

                # add to the lines
                line1 += "{:-6.2f} ".format(neuron.value)                
                con_tuple = "({:-1},{:-1})".format( neuron.connections_to, len(neuron.sending) )
                line2 += "{:>6} ".format(con_tuple)


            # finished data for start of lines
            neuron_count             = len(self.layers[i])
            average_connections_from = round( (total_connections_from / neuron_count), 1)

            # for ending line stats
            total_neurons     += len(self.layers[i])
            total_connections += total_connections_from

            line1_prefix = "{:>4} | ".format(neuron_count)
            line2_prefix = "{:>4} | ".format(average_connections_from)


            print( (line1_prefix + line1) )
            print( (line2_prefix + line2) )
            print("      ")

        print()
        print("{} layers, {} neurons, {} connections".format( len(self.layers), total_neurons, total_connections))














    # Printing Visual
    # ----------------

    def print3(self):
        '''
        format:

        # neurons, average from |    final value        
        ----------------------- |    --------------  ...
        (total to, total from)  |    (# to, # from)   


                       FGM    FGA   FGM3   FGA3    FTM    FTA   RDif    Ast     TO    Stl    Blk     PF
         12, 3.8 |    -3.25   4.71  -2.62  -1.00   5.79   8.50  -1.38  -2.62   0.29   0.08   0.58   3.67 
        ( 0, 45) |    (0,4)  (0,6)  (0,3)  (0,4)  (0,3)  (0,4)  (0,2)  (0,4)  (0,6)  (0,4)  (0,3)  (0,2) 

          9, 3.0 |     0.66   1.00  -0.82  -1.00  -1.00   0.82  -1.00   1.00  -0.99 
        (45, 27) |    (6,3)  (5,3)  (2,3)  (7,2)  (4,5)  (5,1)  (7,3)  (4,3)  (5,4) 

          8, 3.5 |     0.18  -0.95  -0.15   0.90   0.15   0.93  -1.00   0.03 
        (27, 28) |    (1,2)  (3,6)  (4,3)  (3,2)  (2,3)  (4,5)  (7,4)  (3,3) 

          9, 1.0 |    -0.82  -0.94  -0.40   0.18   0.69   0.66  -0.65  -0.54   0.92 
        (28,  9) |    (4,1)  (5,1)  (1,1)  (1,1)  (3,1)  (3,1)  (4,1)  (5,1)  (2,1) 

          1, 0.0 |    -0.91 
        ( 9,  0) |    (9,0)

        '''
        print("format:")
        print("")
        print("# neurons, average from |    final value        ") 
        print("----------------------- |    --------------  ...")
        print("(total to, total from)  |    (# to, # from)   \n\n")

        print("               FGM    FGA   FGM3   FGA3    FTM    FTA   RDif    Ast     TO    Stl    Blk     PF")
             # 12, 4.1 |    -3.25   4.71  -2.62  -1.00   5.79   8.50  -1.38  -2.62   0.29   0.08   0.58   3.67 
             #( 0, 49) |    (0,4)  (0,4)  (0,5)  (0,3)  (0,4)  (0,4)  (0,5)  (0,5)  (0,4)  (0,3)  (0,4)  (0,4)

        for i in range(len(self.layers)):
            # prep for the printed lines
            line1 = "" 
            line2 = ""

            # prep for data
            total_connections_to   = 0
            total_connections_from = 0
            for neuron in self.layers[i].values():
                # tracking data for later
                total_connections_to   += neuron.connections_to
                total_connections_from += len(neuron.sending)

                # add to the lines
                line1 += "{:-6.2f} ".format(neuron.value)                
                con_tuple = "({:-1},{:-1})".format( neuron.connections_to, len(neuron.sending) )
                line2 += "{:>6} ".format(con_tuple)


            # finished data for start of lines
            neuron_count             = len(self.layers[i])
            average_connections_from = round( (total_connections_from / neuron_count), 1)

            #                 12, 4.0 |   -
            line1_prefix = " {:>2},{:>4} |   ".format(neuron_count, average_connections_from)
            #               (  0, 48) |   -
            line2_prefix = "({:>2},{:>3}) |   ".format(total_connections_to, total_connections_from)


            print( (line1_prefix + line1) )
            print( (line2_prefix + line2) )
            print()







    def print2(self):
        print("format  ->  final value    ") 
        print("            -------------- ")
        print("            (# to, # from)  \n")

        #print("   FGM    FGA   FGM3   FGA3    FTM    FTA     OR     DR    Ast     TO    Stl    Blk     PF")
              #  -0.94   1.00  -1.00  -0.89   1.00   1.00   0.85   0.66  -0.99  -0.84   0.63   0.04   1.00 
              #  (1,6)  (1,3)  (1,2)  (1,3)  (1,5)  (1,5)  (1,7)  (1,4)  (1,6)  (1,3)  (1,3)  (1,7)  (1,5) 

        print("   FGM    FGA   FGM3   FGA3    FTM    FTA   RDif    Ast     TO    Stl    Blk     PF")
             #  -3.25   4.71  -2.62  -1.00   5.79   8.50  -1.38  -2.62   0.29   0.08   0.58   3.67 
             #  (0,6)  (0,3)  (0,3)  (0,3)  (0,3)  (0,6)  (0,3)  (0,4)  (0,3)  (0,6)  (0,4)  (0,4) 

        for i in range(len(self.layers)):
            neuron_count     = len(self.layers[i])
            connections_from = 0

            line1 = "" # " " * 16
            line2 = "{:>3}".format(connections_from_count) # " " * 16
            for neuron in self.layers[i].values():
                line1 += "{:-6.2f} ".format(neuron.value)
                con_tuple = "({:-1},{:-1})".format( neuron.connections_to, len(neuron.sending) )
                line2 += "{:>6} ".format(con_tuple)
            print(line1)
            print(line2)
            print()


















# end