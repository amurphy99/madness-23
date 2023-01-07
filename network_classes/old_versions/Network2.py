#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:31:02 2022

@author: andrewmurphy
"""

import random
import copy





class Neuron:
    '''
    Neuron: -> bias, value, input_connections, value_calculated
    -------------
    bias     = float between a range given by the agent's species
    value    = the value of the neuron if it has been resolved/calculated
    inputs   = dictionary of the other neurons this takes as inputs and the weights they are given
    resolved = bool to check if it has already been resolved, so no need to recalculate its value
    
    Functions:
    -------------
    calculate_value -> return the value of this Neuron
    mutate          -> mutate based on the species
    print           -> print out the neurons info with nice formatting
    
    later on..?
        - save to txt
        - load to txt
    '''
    
    def __init__(self, species, inputs_list, value=0, resolved=False):
        # species = bias_range, weight_range, bias_mutation, weight_mutation
    
        inputs = {} # {Neuron : weight}
        for neuron in inputs_list:
            inputs[neuron] = random.uniform( -species.weight_range, species.weight_range )
        
        self.bias     = random.uniform( -species.bias_range, species.bias_range )
        self.value    = value    # 0
        self.inputs   = inputs
        self.resolved = resolved # False
        
        
        
    def quick_string(self):
        bias_   = "bias   = {:<5}".format( round(self.bias,3) )
        weights = "inputs = {"
        
        keys = list(self.inputs.keys())
        if len(keys) == 0: weights += "  "
        for i in range(len(keys)):
            weights += "{:>5}, ".format(round(self.inputs[keys[i]],2))
        
        return (bias_ + "\n" + weights[:-2] + "}")
        
        
        
        

    def calculate_value(self, sigmoid=True):
        # if the value has already been calculated, we can just return it, no need to redo
        if self.resolved:
            return self.value
        
        else:
            self.value = self.bias
            for connection in self.inputs:
                self.value += self.inputs[connection] * connection.calculate_value()
            
            
            # sigmoid -------- -------- -------- --------
            # what is the actual calculation they do? to not just use 0 or 1
            # number of connections * limit = max possible
            # so could do abs(value) / max possible
            if sigmoid:
                if self.value > 0: self.value =  1
                if self.value < 0: self.value = -1
            # -------- -------- -------- -------- --------
            
            self.resolved = True
            return self.value
        
        
        
    def mutate(self, species, limit=True):
        # species = bias_range, weight_range, bias_mutation, weight_mutation
        
        # limit to no bigger than 2
        if limit:
            lim = 2.0
            # connections
            for connection in self.inputs:
                new = self.inputs[connection] + (random.uniform( -species.weight_mutation, species.weight_mutation ))
                if new >  lim: new =  lim
                if new < -lim: new = -lim    
                self.inputs[connection] = new
            
            # bias
            new = self.bias + (random.uniform( -species.bias_mutation, species.bias_mutation ))
            if new >  lim: new =  lim
            if new < -lim: new = -lim
            self.bias      = new
            
            # others
            self.value     = 0
            self.resolved  = False
        
        
        else:
            for connection in self.inputs:
                self.inputs[connection] += (random.uniform( -species.weight_mutation, species.weight_mutation ))

            self.bias     += (random.uniform( -species.bias_mutation, species.bias_mutation ))
            self.value     = 0
            self.resolved  = False
            
            
    
    def reset(self, species):
        # reset bias
        self.bias = random.uniform( -species.bias_range, species.bias_range )
        # reset each connection weight
        for connection in self.inputs:
            self.inputs[connection] = random.uniform( -species.weight_range, species.weight_range )
        














class Species:
    '''
    Species: -> bias_range, weight_range, bias_mutation, weight_mutation
    -------------
    bias_range      = range of values that the bias can be initialized to
    weight_range    = range of values that the weights can be initialized to
    bias_mutation   = range of values that the bias can be mutated by
    weight_mutation = range of values that the weights can be mutated by
    
    (later on...)
    chances for neurons to be added/deleted
    chances for connections to be added/deleted
    
    
    Functions:
    -------------
    print -> print out the neurons info with nice formatting
    
    (just print for now, later on will probably mutate these as well)
    '''
    
    def __init__(self, bias_range, weight_range, bias_mutation, weight_mutation):
        # neuron ranges
        self.bias_range      = bias_range
        self.weight_range    = weight_range
        self.bias_mutation   = bias_mutation
        self.weight_mutation = weight_mutation
        
        # layer information
        self.connection_chance      = 0.5 
        self.connection_chance_rate = 10 
        self.neurons_per_new_layer  = 12
        
        
        
    def quick_string(self):
        bias   = "bias:   {} - {:>.2}".format(self.bias_range,   self.bias_mutation  )
        weight = "weight: {} - {:>.2}".format(self.weight_range, self.weight_mutation)
        return (bias + "\n" + weight)
    





class Agent:
    '''
    (right now i have no oversight on layers)
    or.....
    when a layer is added, nodes in that new layer can take inputs from ANY of the previous layers
    
    layer = list of neurons
    layers = list of the layers
    
    
    
    Agent:
    -------------
    species
    layers
    final node
    
    
    Functions:
    -------------
    new_inputs
    calculate_value
    
    mutate
    
    '''
    
    def __init__(self, species, initial_inputs):
        self.species = species
        self.layers  = [initial_inputs]
        
        self.final_neuron = Neuron(self.species, self.layers[-1])
        
        
    
    def quick_string(self):
        underline = '-' * 14
        
        species_header      = "Species: (init-mut)"
        species_body        = self.species.quick_string()
        
        layers_header       = "Layers: (layer-neurons)"
        layers_body         = ""
        for i in range(len(self.layers)):
            layers_body += " {}-{} ".format(i,len(self.layers[i]))
        
        final_neuron_header = "Final Neuron:"
        final_neuron_body_1 = "bias:       {:>5.2}".format(self.final_neuron.bias)
        final_neuron_body_2 = "connections:{:>2}"   .format(len(self.final_neuron.inputs))
        final_neuron_body   = final_neuron_body_1 + "\n" + final_neuron_body_2
        
        species_whole       = species_header      + "\n" + underline + "\n" + species_body      + "\n"      
        layers_whole        = layers_header       + "\n" + underline + "\n" + layers_body       + "\n"    
        final_neuron_whole  = final_neuron_header + "\n" + underline + "\n" + final_neuron_body
        
        return ( species_whole + "\n" + layers_whole + "\n" + final_neuron_whole )
    
    
    
    def update_inputs(self, new_inputs):
        for i in range(len(self.layers[0])):
            self.layers[0][i].value = new_inputs[i]
            
    
    def calculate_value(self):
        return self.final_neuron.calculate_value()
    
    
    def mutate(self):
        # can't mutate the input layer
        if len(self.layers) > 1:
            for i in range(1,len(self.layers)):
                for neuron in self.layers[i]:
                    neuron.mutate(self.species)
        
        # mutate final neuron
        self.final_neuron.mutate(self.species)
        
        
        
    def reset_evaluations(self):
        # can't reset the input layer
        if len(self.layers) > 1:
            for i in range(1,len(self.layers)):
                for neuron in self.layers[i]:
                    neuron.value    = 0
                    neuron.resolved = False
        
        # mutate final neuron
        self.final_neuron.value    = 0
        self.final_neuron.resolved = False
        
        
        
        
        
    def add_layer(self):
        # things needed from species:
        # connection_chance, connection_chance_rate, neurons_per_new_layer


        # creating the each new neuron in the layer
        # --------------------------------------------
        new_layer         = [self.final_neuron]          # keeping the final neuron as the first of the new layer
        neurons_to_create = self.species.neurons_per_new_layer

        # individual neurons
        for i in range(neurons_to_create):
            # determine connections
            connections = []
            chance      = self.species.connection_chance        # 0.50
            chance_rate = self.species.connection_chance_rate   # 2

            for j in range((len(self.layers)-1), -1, -1):
                # looping backwards through each layer 
                for neuron in self.layers[j]:
                    if chance > random.uniform(0,1):
                        connections.append(neuron)            
                # decrement the connection chance for the next layer
                chance = chance / chance_rate

            # create new neuron from those connections and add it to the new layer
            new_layer.append( Neuron(self.species, connections) )



        # officially add the new layer and remake the final neuron
        # --------------------------------------------
        self.layers.append(new_layer)
        self.final_neuron = Neuron(self.species, self.layers[-1])
        
        
        
        
    def reset(self):
        for i in range(len(self.layers)):
            for neuron in self.layers[i]:
                neuron.reset(self.species)






















class Population:
    '''
    Population:
    -------------    
    inputs
    species_list
    agents_list
    
    agent_ranks
    species_ranks
    evolution_steps_count
    
    
    Functions:
    -------------
    mutate
    
    '''
    
    def __init__(self, initial_inputs, given_species, max_score=-1, agents_per_species=1):
        self.max_score    = max_score
        self.inputs       = initial_inputs
        self.species_list = given_species
        
        # creating all agents
        self.agents_list = []
        for species in self.species_list:
            for i in range(agents_per_species):
                self.agents_list.append(Agent(species, self.inputs))
        
        # "ranks" for each agent (will count # of correct evalutations)
        self.agent_ranks      = [0 for i in range(len(self.agents_list))]
        
        # "ranks" for each species (will count # agents who survive in its slot)
        self.species_ranks    = [0 for i in range(len(self.species_list))]
        
        # initializing a list of lifespans for each agent
        self.agent_lifespans  = [0 for i in range(len(self.species_list))]
        self.agent_prev_score = [0 for i in range(len(self.species_list))] 
        
        # number to keep track of how many times the population has been mutated
        self.evolution_steps_count = 0
    
    
    
    # misc helpers
    # ------------------------------------------------------------------------
    def mutate_all(self):
        for agent in self.agents_list:
            agent.mutate()
    
    def update_inputs(self, new_inputs):
        for i in range(len(self.inputs)):
            self.inputs[i].value = new_inputs[i]
            
        for agent in self.agents_list:
            agent.update_inputs(new_inputs)
                   
    def reset_evaluations(self):
        for agent in self.agents_list:
            agent.reset_evaluations()
            
    def reset_ranks(self):
        for i in range(len(self.agent_ranks)):
            self.agent_ranks[i] = 0
            
            
            
    # testing
    # ------------------------------------------------------------------------
    
    def test_agents(self, input_combos, answer_key):
        self.reset_ranks()
        # input_combos = [ [1,1], [1,0], [0,1], [0,0] ]
        # answer_key   = [ 1,     0,     0,     0     ]
        
        for i in range(len(input_combos)):
            self.update_inputs( input_combos[i] )
            self.reset_evaluations()
            
            for j in range(len(self.agents_list)):
                evaluation = self.agents_list[j].calculate_value()
                
                if answer_key[i] == 1 and evaluation > 0: self.agent_ranks[j] += 1
                if answer_key[i] == 0 and evaluation < 0: self.agent_ranks[j] += 1
                    
        for i in range(len(self.agent_ranks)):
            #if self.agent_ranks[i] == self.max_score and self.species_ranks[i] == 0:
                #self.species_ranks[i] = self.evolution_steps_count
                
            if self.agent_ranks[i] > self.agent_prev_score[i] and self.agent_prev_score[i] != 0:
                self.agent_prev_score[i] = self.agent_ranks[i]
                self.species_ranks[i] += 1
            
            #elif self.agent_ranks[i] == self.agent_prev_score[i]:
                #self.agent_lifespans[i] += 1
                
            else:
                self.agent_prev_score[i] = self.agent_ranks[i]
                #self.agent_lifespans [i] = 0
            
        return self.agent_ranks.copy()
    
    
    
    
    
    # SECOND VERSION
    # ------------------------------------------------------------------------------------
    def agent_split_v2(self, split_ratio=0.5):
        
        # calculating cutoff specifics
        # -----------------------------
        survivors = round(len(self.agent_ranks) * split_ratio)
        cloning   = len(self.agent_ranks) - survivors

        
        # formatting scores
        # -----------------------------
        results = self.agent_ranks.copy() # [2, 1, 3, 2, 1, 1, 3, 2, 1, 3]
        ranked  = {}                      # {2:3, 1:4, 3:3}
        for i in range(len(results)):
            if results[i] in ranked: ranked[results[i]].append(i)        
            else:                    ranked[results[i]] = [i]

        scores = list(ranked.keys()) # [2,1,3]
        scores.sort()                # [1,2,3]
        scores.reverse()             # [3,2,1]

        
        # create list of indicis to survive
        # ------------------------------------------
        
        # (all top ranked agents automatically get to survive)
        survived = ranked[scores[0]]
        if len(survived) > survivors:
            survivors = 0
            cloning   = len(results) - len(survived)
        else:
            survivors -= len(survived)
            
        # survived = [] (used to be this but now changed ot make sure all top ranks survive)
        for i in range(1,len(scores)):
            if survivors <= 0: break

            queue = ranked[scores[i]].copy()
            random.shuffle(queue)

            while len(queue) > survivors:
                index = random.randrange(len(queue))
                queue.pop(index)

            survived  += queue
            survivors -= len(queue)


        # create list of indicis to be cloned
        # ------------------------------------------
        cloned = []
        while cloning > 0:
            if len(survived) >= cloning:
                cloned  += survived[:cloning]
                cloning  = 0
            else:
                cloned  += survived
                cloning -= len(survived)


        # create list of indicis to be eliminated
        # ------------------------------------------
        eliminated = []
        for i in range(len(results)):
            if i not in survived:
                eliminated.append(i)


        # return all 3
        return [survived, cloned, eliminated]
        # ------------------------------------------------------------------------------------
    
    
    
    
    def evolution_step(self, split_ratio=0.5):
        # [survived, cloned, eliminated]
        #splits     = self.agent_split(split_ratio)
        splits     = self.agent_split_v2(split_ratio)
        survived   = splits[0]
        cloned     = splits[1]
        eliminated = splits[2]

        # adding to the scores of the species of surviving agents
        for index in survived:
            #self.species_ranks  [index] += 1
            self.agent_lifespans[index] += 1


        # cloning step
        # ----------------------------
        for i in range(len(eliminated)):
            # reset lifespan
            self.agent_lifespans [ eliminated[i] ] = 0
            self.agent_prev_score[ eliminated[i] ] = self.agent_prev_score[ cloned[i] ]
            
            # deepcopy and then change the species
            new_clone         = copy.deepcopy(self.agents_list[ cloned[i] ])
            new_clone.species = self.species_list[ eliminated[i] ]

            self.agents_list[ eliminated[i] ] = new_clone


        # mutation step
        # ----------------------------
        self.mutate_all()
        self.evolution_steps_count += 1
        
        # retiring agents with no improvment in 50 epochs
        # self.retire_agents(10)
        
        
        
    def retire_agents(self, age_limit):
        # dont retire if they are right
        
        for i in range(len(self.agent_lifespans)):
            if self.agent_lifespans[i] > age_limit and self.agent_prev_score[i] < self.max_score:
                self.agents_list     [i].reset()
                self.agent_lifespans [i] = 0
                self.agent_prev_score[i] = 0
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                