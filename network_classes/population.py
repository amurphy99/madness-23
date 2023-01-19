import random
import copy
import threading
from math import log
from time import time

from .neuron import Neuron
from .agent  import Agent




class Population:
	'''


	'''
	def __init__(self, num_inputs, num_agents=10):
		
		# Agents
		# -------
		self.agents = []
		self.scores = []
		self.base_scores = []
		self.test = []
		
		# for confidence scoring
		# -----------------------
		# [              0,                   1,                   2,               3]
		# [confident_right, not_confident_right, not_confident_wrong, confident_wrong]
		self.confidence_scores = []


		for i in range(num_agents):
			self.agents.append( Agent(num_inputs) )
			self.scores.append( 0 )
			self.base_scores.append( 0 )
			self.confidence_scores.append( [0,0,0,0] )
			self.test.append([])


		# Statistics
		# -----------
		self.total_steps = 0
		self.best_score  = 0

		# evolution timing
		self.time_cloning = 0
		self.time_mutating = 0

		self.time_splitting_population = 0
		self.time_creating_score_dict  = 0
		self.time_selecting_survivors  = 0 

		# scoring timing
		self.num_calculations = 0
		self.time_calculating_values = 0
		self.time_sending_values 	 = 0

		self.time_setting_inputs     = 0
		self.time_tanh_scoring		 = 0

		self.time_threading 		= 0
		self.time_creating_threads 	= 0
		self.time_starting_threads	= 0
		self.time_joining_threads	= 0






#########################################################################################################################
#													Testing Functions													#
#########################################################################################################################


	def reset_all_scores(self):
		for i in range(len(self.agents)):
			self.scores[i] 				= 0
			self.base_scores[i] 		= 0
			self.test[i]              	= []
			self.confidence_scores[i] 	= [0,0,0,0]


	def logic_scoring(self, agent_index, output, solution):
		# set it to 0 or 1
		if output > 0: 
			result = 1
		else:
			result = 0

   		# score it
		if result == solution:
			self.base_scores[agent_index] += 1


	# confidence scoring
	# -------------------
	def logic_to_conf(self):
		for i in range(len(self.scores)):
			conf_total  = self.confidence_scores[i][0] *  7
			conf_total += self.confidence_scores[i][1] *  4
			conf_total += self.confidence_scores[i][2] * -2
			conf_total += self.confidence_scores[i][3] * -8

			self.scores[i] = conf_total
			if self.scores[i] == 0: print("ERROR in logic_to_conf()")



	def confidence_scoring(self, agent_index, output, solution):
		# score values
		# -------------
		confident_right     =  7
		not_confident_right =  4
		not_confident_wrong = -2
		confident_wrong     = -8


		# assigning scores
		# -----------------

		# confident_right
		if (solution > 0 and output > 0.5) or (solution <= 0 and output < -0.5):
			self.confidence_scores[agent_index][0] += 1
			#self.scores           [agent_index]    += confident_right

		# not confident right
		elif (solution > 0 and output > 0) or (solution <= 0 and output < 0):
			self.confidence_scores[agent_index][1] += 1
			#self.scores           [agent_index]    += not_confident_right

		# not confident wrong
		elif (solution > 0 and output > -0.5) or (solution <= 0 and output < 0.5):
			self.confidence_scores[agent_index][2] += 1
			#self.scores           [agent_index]    += not_confident_wrong

		# confident wrong
		elif (solution > 0 and output < -0.5) or (solution <= 0 and output > 0.5):
			self.confidence_scores[agent_index][3] += 1
			#self.scores           [agent_index]    += confident_wrong

		else: print("idk")



	def tanh_confidence_scoring(self, agent_index, output, solution):
		# score values
		# -------------
		abs_output = abs(output)
		if abs_output >= 0.90:
			post_tanh = (1/2) * log((1+0.90)/(1-0.90)) 
		else:
			post_tanh = (1/2) * log((1+abs_output)/(1-abs_output)) 

		# try increasing it
		correct_score   = 1.40**post_tanh
		incorrect_score = 1.60**post_tanh

		# assigning scores
		# -----------------
		# right
		if (solution > 0 and output > 0) or (solution <= 0 and output <= 0):
			#self.base_scores[agent_index] += 1
			self.scores[agent_index]      += correct_score 
			#self.test[agent_index].append(correct_score + 1)
			# conf list
			#if abs_output > 0.5: 	self.confidence_scores[agent_index][0] += 1
			#else: 					self.confidence_scores[agent_index][1] += 1

		# wrong
		elif (solution > 0 and output <= 0) or (solution <= 0 and output > 0):
			self.scores[agent_index] -= incorrect_score
			#self.test[agent_index].append(-incorrect_score)
			# conf list
			#if abs_output > 0.5: 	self.confidence_scores[agent_index][3] += 1
			#else: 					self.confidence_scores[agent_index][2] += 1





	# normal method
	# --------------
	def test_agents_double(self, inputs, solutions, show_scores=False):
		'''
		# if team 1 was the winner:
		# first test = positive
		# second test = negative
		# first answer = positive
		# so: (first_test - second_test) should be > 0

		# if team 2 was the winner:
		# first test = negative,
		# second test = positive
		# first answer = negative
		# so: (first_test - second_test) should be < 0

		'''
		# test 1 agent at a time on all data
		for i in range(len(self.agents)):
			#self.scores[i]            = 0 # just temporary, resetting scores here
			#self.base_scores[i]       = 0
			#self.test[i]              = []
			#self.confidence_scores[i] = [0,0,0,0]


			if self.base_scores[i] == 0: # only re-test if the score was reset

				current = 0
				for j in range( len(inputs)//2 ):
					
					# FIRST way around
					self.agents[i].set_inputs(inputs[current])
					first_test, calc_time, send_time = self.agents[i].calculate_value()
					#self.time_calculating_values += calc_time
					#self.time_sending_values     += send_time

					# SECOND way around
					self.agents[i].set_inputs(inputs[current+1])
					second_test, calc_time, send_time = self.agents[i].calculate_value()
					#self.time_calculating_values += calc_time
					#self.time_sending_values     += send_time

					
					# final decision (for the final decision we will use the first way around)
					output = (first_test - second_test)

					# score the agents final evaluation
					self.tanh_confidence_scoring(i, output, solutions[current])
					self.confidence_scoring(i, output, solutions[current])
					self.logic_scoring(i, output, solutions[current])           # normal scoring

					# iterate j an extra value since we use two per loop
					current += 2

				
				if i == 0 and show_scores:
					uniques = {}
					for score in self.test[i]:
						rounded = round(score, 1)
						if rounded in uniques: uniques[rounded] += 1
						else:                  uniques[rounded]  = 1 
					
					for entry in uniques.keys():
						print( "{:>6} - {}".format(entry, uniques[entry]) )





	def timed_test_agents_double(self, inputs, solutions):
		threading_time = time()
		# test 1 agent at a time on all data
		for i in range(len(self.agents)):
			

			if self.base_scores[i] == 0: # only re-test if the score was reset

				current = 0
				for j in range( len(inputs)//2 ):
					
					# FIRST way around
					# -----------------
					set_inputs_time = time()
					self.agents[i].set_inputs(inputs[current])
					self.time_setting_inputs += (time()-set_inputs_time)

					first_test, calc_time, send_time = self.agents[i].calculate_value()
					self.time_calculating_values += calc_time
					self.time_sending_values     += send_time


					# SECOND way around
					# ------------------
					set_inputs_time = time()
					self.agents[i].set_inputs(inputs[current+1])
					self.time_setting_inputs += (time()-set_inputs_time)

					second_test, calc_time, send_time = self.agents[i].calculate_value()
					self.time_calculating_values += calc_time
					self.time_sending_values     += send_time

					
					# Scoring
					# --------
					tanh_scoring_time = time()
					# final decision (for the final decision we will use the first way around)
					output = (first_test - second_test)
					# score the agents final evaluation
					self.tanh_confidence_scoring(i, output, solutions[current])
					self.time_tanh_scoring += (time()-tanh_scoring_time)


					# iterate j an extra value since we use two per loop
					current += 2

				self.num_calculations += 1

		self.time_threading += (time()-threading_time)









	# threading methods
	# -----------------------------
	def test_agents_thread_double(self, agent_index, inputs, solutions):
		'''
		# if team 1 was the winner:
		# first test = positive
		# second test = negative
		# first answer = positive
		# so: (first_test - second_test) should be > 0

		# if team 2 was the winner:
		# first test = negative,
		# second test = positive
		# first answer = negative
		# so: (first_test - second_test) should be < 0

		'''
		# loop through given data
		current = 0
		for j in range( len(inputs)//2 ):
									
			# FIRST way around
			set_inputs_time = time()
			self.agents[agent_index].set_inputs(inputs[current])
			self.time_setting_inputs += (time()-set_inputs_time)

			first_test, calc_time, send_time = self.agents[agent_index].calculate_value()
			self.time_calculating_values += calc_time
			self.time_sending_values     += send_time

			# SECOND way around
			set_inputs_time = time()
			self.agents[agent_index].set_inputs(inputs[current+1])
			self.time_setting_inputs += (time()-set_inputs_time)

			second_test, calc_time, send_time = self.agents[agent_index].calculate_value()
			self.time_calculating_values += calc_time
			self.time_sending_values     += send_time

			# final decision
			# (for the final decision we will use the first way around)
			output = (first_test - second_test)

		    # score the agents final evaluation
	    	# ----------------------------------
			self.tanh_confidence_scoring(agent_index, output, solutions[current])

			# iterate j an extra value since we use two per loop
			current += 2



	def test_agents_threading_double(self, inputs, solutions):
		threading_time = time()

		# keep all threads in a list
		threads = []

		# create all threads and store them
		creating_threads_time = time()
		for i in range(len(self.agents)):
			if self.scores[i] == 0: # only re-test if the score was reset
				agent_thread = threading.Thread(target = self.test_agents_thread_double, args = (i, inputs, solutions))
				threads.append(agent_thread)
		self.time_creating_threads += (time()-creating_threads_time)


		# Start all threads
		starting_threads_time = time()
		for single in threads:
			single.start()
		self.time_starting_threads += (time()-starting_threads_time)


		# Wait for all of them to finish
		joining_threads_time = time()
		for wait in threads:
			wait.join()
			self.num_calculations += 1
		self.time_joining_threads += (time()-joining_threads_time)


		self.time_threading += (time()-threading_time)

		self.best_score = max(self.scores)




#########################################################################################################################
#													Mutation Functions													#
#########################################################################################################################


	def split_population(self, survivor_percentage=0.5):
		split_population_time = time()

		# survivor goal
		needed_survivors = int(len(self.agents) * survivor_percentage)

		# dictionary for higher scorers
		score_dictionary_time = time()
		score_dictionary = {}
		for i in range(len(self.scores)):

			score = self.scores[i]
			if score in score_dictionary:
				score_dictionary[score].append(i)
			else:
				score_dictionary[score] = [i]

		# sort high scores
		unique_scores = list( score_dictionary.keys() )
		unique_scores.sort(reverse=True)
		self.time_creating_score_dict += (time()-score_dictionary_time)

		# select survivors
		selecting_surviors_time = time()
		survivors = []
		remaining = needed_survivors - len(survivors)
		for i in range(len(unique_scores)):
			# check if done
			remaining = needed_survivors - len(survivors)
			if remaining == 0: break

			# first top score
			scorers = score_dictionary[unique_scores[i]].copy()
			if len(scorers) <= remaining:
				survivors += scorers
			else:
				# need to select # of remaining needed from the scorers list
				random.shuffle(scorers)
				sample = scorers[:remaining]
				survivors += sample
		self.time_selecting_survivors += (time()-selecting_surviors_time)

		self.time_splitting_population += (time()-split_population_time)
		return survivors



	def create_clones(self, survivor_indicis):
		# create new list of agents
		new_agents = list(range( len(self.agents) ))

		# create clones and reassign originals
		clone_time = time()

		clones = []
		for index in survivor_indicis:
			new_agents[index]         = self.agents[index]   # original
			clones.append(copy.deepcopy(self.agents[index])) # clone

		self.time_cloning += (time() - clone_time)
 
		# fill empty indicis with the clones
		for i in range(len(new_agents)):
			if type(new_agents[i]) == int:
				# reset the score of the clone
				self.confidence_scores[i] = [0,0,0,0] 
				self.base_scores[i]       = 0
				self.scores[i]            = 0 
				self.test[i] 			  = []
				
				# assign and mutate new clone
				new_agents[i] = clones.pop(0)

				mutation_time = time()
				new_agents[i].mutate()
				self.time_mutating += (time()-mutation_time)


		# assign the new agents list
		self.agents = new_agents
		


	def evolution_step(self, reset_confidence_scores=True):
		survivors = self.split_population()
		self.create_clones(survivors)
		









	























































