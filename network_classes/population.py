import random
import copy
import threading
from math import log


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
		


	# Testing Functions
	# ------------------
	# (this is one of the places where the threading could be used)
	def test_agents(self, inputs, solutions, max_steps=100, max_score=-1):

		# test 1 agent at a time on all data
		for i in range(len(self.agents)):
			if self.scores[i] == 0:
				for j in range(len(inputs)):
					
					# set inputs and calculate output
					self.agents[i].set_inputs(inputs[j])
					output = self.agents[i].calculate_value()

	    	    	# score if it is right or now
					self.logic_scoring(i, output, solutions[j])

		#self.best_score = max(self.scores)


	def logic_scoring(self, agent_index, output, solution):
		# set it to 0 or 1
		if output > 0: 
			result = 1
		else:
			result = 0

   		# score it
		if result == solution:
			self.base_scores[agent_index] += 1




	# version 2
	# (for when testing a game both ways)
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
			self.scores[i]            = 0 # just temporary, resetting scores here
			self.base_scores[i]       = 0
			self.test[i]              = []
			self.confidence_scores[i] = [0,0,0,0]


			if self.scores[i] == 0: # only re-test if the score was reset

				current = 0
				for j in range( len(inputs)//2 ):
					
					# FIRST way around
					self.agents[i].set_inputs(inputs[current])
					first_test = self.agents[i].calculate_value()

					# SECOND way around
					self.agents[i].set_inputs(inputs[current+1])
					second_test = self.agents[i].calculate_value()

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
		if abs_output >= 0.99:
			post_tanh = 2.8
		else:
			post_tanh  = (1/2) * log((1+abs_output)/(1-abs_output)) 

		# try increasing it
		post_tanh = 1.25**post_tanh

		# assigning scores
		# -----------------

		# right
		if (solution > 0 and output > 0) or (solution <= 0 and output <= 0):
			self.scores[agent_index] += post_tanh
			self.test[agent_index].append(post_tanh)

		# wrong
		elif (solution > 0 and output <= 0) or (solution <= 0 and output > 0):
			self.scores[agent_index] -= post_tanh
			self.test[agent_index].append(-post_tanh)






	# For mutation
	# -------------

	def split_population(self, survivor_percentage=0.5):
		# survivor goal
		needed_survivors = int(len(self.agents) * survivor_percentage)

		# dictionary for higher scorers
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

		# select survivors
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

		return survivors



	def create_clones(self, survivor_indicis):
		# create new list of agents
		new_agents = list(range(10))

		# create clones and reassign originals
		clones = []
		for index in survivor_indicis:
			# original
			new_agents[index] = self.agents[index]

			# clone + mutate the clone
			clone = copy.deepcopy(self.agents[index])
			clone.mutate()
			clones.append(clone)

		# fill empty indicis with the clones
		for i in range(len(new_agents)):
			if type(new_agents[i]) == int:
				self.scores[i] = 0 # reset the score of the clone
				self.confidence_scores[i] = [0,0,0,0] 
				new_agents[i]  = clones.pop(0)

		# assign the new agents list
		self.agents = new_agents
		


	def evolution_step(self, reset_confidence_scores=True):
		survivors = self.split_population()
		self.create_clones(survivors)
		
		for i in range(len(self.scores)):
			# only reset the scores for the non clones
			if i not in survivors:
				self.scores[i]      = 0
				self.base_scores[i] = 0
				if reset_confidence_scores:
					self.confidence_scores[i] = [0,0,0,0]





	# trying threading
	# -----------------

	def test_agents_thread(self, agent_index, inputs, solutions):
		# loop through given data
		for j in range(len(inputs)):
									
			# set inputs and calculate output
			self.agents[agent_index].set_inputs(inputs[j])
			output = self.agents[agent_index].calculate_value()

	    	# score if it is right or now
			self.logic_scoring(agent_index, output, solutions[j])


	def test_agents_threading(self, inputs, solutions):
		# keep all threads in a list
		threads = []

		# create all threads and store them
		for i in range(len(self.agents)):
			if self.scores[i] == 0: # only re-test if the score was reset
				agent_thread = threading.Thread(target = self.test_agents_thread, args = (i, inputs, solutions))
				threads.append(agent_thread)

		# Start all threads
		for single in threads:
			single.start()

		# Wait for all of them to finish
		for wait in threads:
			wait.join()

		self.best_score = max(self.scores)




	# double testing method
	# ----------------------
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
			self.agents[agent_index].set_inputs(inputs[current])
			first_test = self.agents[agent_index].calculate_value()

			# SECOND way around
			self.agents[agent_index].set_inputs(inputs[current+1])
			second_test = self.agents[agent_index].calculate_value()

			# final decision
			# (for the final decision we will use the first way around)
			output = (first_test - second_test)

		    # score the agents final evaluation
	    	# ----------------------------------
			#self.logic_scoring(agent_index, output, solutions[current])     # normal scoring
			self.tanh_confidence_scoring(agent_index, output, solutions[current])
			#self.confidence_scoring(agent_index, output, solutions[current]) # confidence scoring

			# iterate j an extra value since we use two per loop
			current += 2



	def test_agents_threading_double(self, inputs, solutions):
		# keep all threads in a list
		threads = []

		# create all threads and store them
		for i in range(len(self.agents)):
			if self.scores[i] == 0: # only re-test if the score was reset
				agent_thread = threading.Thread(target = self.test_agents_thread_double, args = (i, inputs, solutions))
				threads.append(agent_thread)

		# Start all threads
		for single in threads:
			single.start()

		# Wait for all of them to finish
		for wait in threads:
			wait.join()

		self.best_score = max(self.scores)





































































