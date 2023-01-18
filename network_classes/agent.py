# for class Agent

import random

from time import time
from sys  import getsizeof

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
        self.starting_neurons     =  int( num_inputs / 1.50 )
        self.starting_connections =  4

        self.bias_range   = 1.0
        self.weight_range = 1.0

        self.max_bias   = 3.0
        self.max_weight = 3.0

        # Species Mutation Info
        # ----------------------
        self.bias_mutation_range   = 0.5
        self.weight_mutation_range = 0.5

        self.bias_mutation_chance   = 0.80
        self.weight_mutation_chance = 0.80
        self.mut_connection_chance  = 0.60
        self.mut_neuron_chance      = 0.30


        # stats
        # ------
        #self.time_deleting = 0
        #self.time_cloning  = 0



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
                    if i == 0:
                        self.new_connection(neuron, (i+1,i+1))                  # only the next
                    else:
                        self.new_connection(neuron, (i+1,i+1))                 # only the next
                        #self.new_connection(neuron, (i+1,len(self.layers)-2))   # all but final layer


        # add connections to the final neuron
        for neuron in self.layers[-2].values():
            neuron.sending.append( [(-1,"final_key"), 1.0] )





#########################################################################################################################
#                                                 Checking Sizes                                                        #
#########################################################################################################################


    def __sizeof__(self):
        total_size = 0
        total_size += getsizeof(self.starting_layers)
        total_size += getsizeof(self.starting_neurons)
        total_size += getsizeof(self.starting_connections)

        total_size += getsizeof(self.bias_range)
        total_size += getsizeof(self.weight_range)

        total_size += getsizeof(self.max_bias)
        total_size += getsizeof(self.max_weight)

        total_size += getsizeof(self.bias_mutation_range)
        total_size += getsizeof(self.weight_mutation_range)

        total_size += getsizeof(self.bias_mutation_chance)
        total_size += getsizeof(self.weight_mutation_chance)
        total_size += getsizeof(self.mut_connection_chance)
        total_size += getsizeof(self.mut_neuron_chance)

        #total_size += getsizeof(self.time_deleting)

        for layer in self.layers:
            keys    = list( layer.keys()   )
            neurons = list( layer.values() )
            for i in range(len(keys)):
                total_size += getsizeof( keys[i]    )
                total_size += getsizeof( neurons[i] )
        
        return total_size



    def size_report(self, all_attributes=True):
        # layer sizes
        # ------------
        all_layers_size = 0
        layers_out = ""
        for j in range(len(self.layers)):
            layer   = self.layers[j]
            keys    = list( layer.keys()   )
            neurons = list( layer.values() )

            layer_size = 0
            for i in range(len(keys)):
                layer_size += getsizeof( keys[i]    )
                layer_size += getsizeof( neurons[i] )

            avg_neuron_size = round( (layer_size/len(layer))/1024 , 2)

            #                   mut_neuron_chance       {:>8}
            #                                           --------    --------    --------
            layers_out      += "layer[{}]                {:>8}    {:>8}    {:>8} \n".format(j, round(layer_size / 1024, 2), len(layer), avg_neuron_size)
            all_layers_size += layer_size


        # total size
        # -----------
        total_size_bytes    = getsizeof(self)
        total_size_kB       = round(total_size_bytes / 1024, 2)
        total_size_mB       = round(total_size_kB    / 1024, 2)
        #                      mut_neuron_chance       {:>8}
        #                                              --------    --------
        total_size_out      = "total_size              {:>8}    {:>8}".format(total_size_kB, total_size_mB)


        # output lines
        # -------------
        if all_attributes:
            out = """
size report
------------
                        bytes   
                        --------
starting_layers         {:>8}
starting_neurons        {:>8}
starting_connections    {:>8}

bias_range              {:>8}
weight_range            {:>8}

max_bias                {:>8}
max_weight              {:>8}

bias_mutation_range     {:>8}
weight_mutation_range   {:>8}

bias_mutation_chance    {:>8}
weight_mutation_chance  {:>8}
mut_connection_chance   {:>8}
mut_neuron_chance       {:>8}

                        kB          neurons     avg size
                        --------    --------    --------
{}

                        kB          mB
                        --------    --------
{}
""".format( getsizeof(self.starting_layers),
            getsizeof(self.starting_neurons),
            getsizeof(self.starting_connections),

            getsizeof(self.bias_range),
            getsizeof(self.weight_range),

            getsizeof(self.max_bias),
            getsizeof(self.max_weight),

            getsizeof(self.bias_mutation_range),
            getsizeof(self.weight_mutation_range),

            getsizeof(self.bias_mutation_chance),
            getsizeof(self.weight_mutation_chance),
            getsizeof(self.mut_connection_chance),
            getsizeof(self.mut_neuron_chance),
            layers_out,
            total_size_out
            )

        else:
            out = """
                        kB          neurons     avg size
                        --------    --------    --------
{}

                        kB          mB
                        --------    --------
{}
""".format( layers_out,
            total_size_out)

        
        print(out)




#########################################################################################################################
#                                             Calculating Values                                                        #
#########################################################################################################################



    # Calculates all of the neuron values
    # ------------------------------------
    def calculate_value(self):
        # starting with the first layer

        time_calculating_values = 0
        time_sending_values     = 0

        for i in range(len(self.layers)):
            layer = self.layers[i]
            for neuron in list(layer.values()):

                # first calculate the final value
                if i != 0:
                    value_calculation_time = time()
                    neuron.calculate_value()
                    time_calculating_values += (time()-value_calculation_time)

                # next send that value to all of the connections
                sending_to_connections_time = time()
                for connection in neuron.sending:
                    address = connection[0]
                    value   = connection[1] * neuron.value
                    self.layers[ address[0] ][ address[1] ].receiving.append(value)
                time_sending_values += (time()-sending_to_connections_time)

        # return final neurons value
        return self.layers[-1]["final_key"].value, time_calculating_values, time_sending_values


    # set new inputs for a calculation
    def set_inputs(self, new_inputs):
        time_setting_inputs = 0
        time_error_checking_inputs = 0
        # quick error checking
        error_check_time = time()
        if len(new_inputs) != len(self.layers[0]):
            print("Error: wrong number of inputs; agent takes {} inputs but received {} values".format( len(self.layers[0]), len(new_inputs) ))
        time_error_checking_inputs += (time()-error_check_time)

        # setting the inputs
        set_inputs_time = 0
        input_layer = list(self.layers[0].values())
        for i in range(len(new_inputs)):
            input_neuron = input_layer[i]
            input_value  = new_inputs[i]
            # broken up just in case i wanted to do somethign with the inputs first
            #input_neuron.receiving.append(input_value)

            input_neuron.value = input_value # for an experiment with non-tanh output from input neurons
        time_setting_inputs += (time()-set_inputs_time)

        return time_error_checking_inputs, time_setting_inputs





#########################################################################################################################
#                                               Mutation Utility                                                        #
#########################################################################################################################



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
            as_int = random.randint(0,99)
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
                #self.new_connection(base_neuron, (layer_index+1,len(self.layers)-2))    # all but final layer




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
        #start_time = time()
        for i in range(0, address[0]):
            for neuron in self.layers[i].values():
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
                    to_delete = neuron.sending.pop(index)
                    del to_delete


        # delete the actual neuron
        # -------------------------
        # self.layers[address[0]]   -> correct dictionary from the layer list
        # address[1]                -> string key to the dictionary for this neuron
        to_delete = self.layers[address[0]].pop(address[1])
        #print(to_delete)
        
        #self.time_deleting += (time()-start_time)





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



    # delete connection
    def del_connection(self, neuron):
        # leave at least 1 connection
        if len(neuron.sending) > 1:
            selected_connection = random.randint(0, len(neuron.sending)-1)
            neuron.sending.pop( selected_connection )




    # Mutating functions
    # -------------------
    def mutate(self, delete_neurons=True):
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
                if self.layers[i][key].connections_to == 0 and delete_neurons:
                    addresses_to_delete.append( [i,key] )

            # delete addresses in a different loop so dictionary doesn't change size during iteration
            for address in addresses_to_delete:
                if len(self.layers[i]) > int(self.starting_neurons // 3):
                    self.delete_neuron( address )


            # add/del neuron
            # -------------------
            if self.mut_neuron_chance > random.uniform(0, 1):
                if 0.5 > random.uniform(0, 1): # 50/50 to add or delete
                    if len(self.layers[i]) < (1.25*len(self.layers[0])):
                        self.add_new_neuron(i)
                else:
                    # * skip a deletion if one was already deleted
                    if len(self.layers[i]) > int(self.starting_neurons // 3) and len(addresses_to_delete) == 0 and delete_neurons:
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
                        # when adding connections, the input layer can ONLY connect to the next one
                        if i == 0: 
                            self.new_connection(neuron, (i+1,i+1)) 
                        else:
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







    






#########################################################################################################################
#                                                     Printing                                                          #
#########################################################################################################################





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







    def print2(self):
        print("format:")
        print("")
        print(" # neurons    |  avg conn weight     ") 
        print(" ------------ |  ---------------     ")
        print(" average from |  (# to, # from)  \n\n")


        total_neurons     = 0
        total_connections = 0
        for i in range(len(self.layers)):
            # prep for the printed lines
            line1 = "" 
            line2 = ""

            # prep for data
            total_connections_to   = 0
            total_connections_from = 0

            layer_neurons = list(self.layers[i].values())
            for j in range(len(layer_neurons)):
                neuron = layer_neurons[j]

                # tracking data for later
                total_connections_to   += neuron.connections_to
                total_connections_from += len(neuron.sending)

                # average weight
                total_connection_weight = 0
                for k in range(len(neuron.sending)):
                    total_connection_weight += abs( neuron.sending[k][1] )

                if len(neuron.sending) == 0: 
                    average_connection_weight = 0
                else: 
                    average_connection_weight = round(total_connection_weight/len(neuron.sending), 2)
                


                if j < 14:
                    # add to the lines
                    line1 += "{:-6.2f} ".format(average_connection_weight)                
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


            #if len(layer_neurons) > 13:


            print( (line1_prefix + line1) )
            print( (line2_prefix + line2) )
            print("      ")

        print()
        print("{} layers, {} neurons, {} connections".format( len(self.layers), total_neurons, total_connections))



















# end