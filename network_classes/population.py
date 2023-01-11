import random
import copy
import threading

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

		for i in range(num_agents):
			self.agents.append( Agent(num_inputs) )
			self.scores.append( 0 				  )


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
			for j in range(len(inputs)):
				
				# set inputs and calculate output
				self.agents[i].set_inputs(inputs[j])
				output = self.agents[i].calculate_value()

    	    	# score if it is right or now
				self.logic_scoring(i, output, solutions[j])

		self.best_score = max(self.scores)



	def logic_scoring(self, agent_index, output, solution):
		# set it to 0 or 1
		if output > 0: 
			result = 1
		else:
			result = 0

   		# score it
		if result == solution:
			self.scores[agent_index] += 1



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
		new_agents = []
		for index in survivor_indicis:
			# original
			new_agents.append(self.agents[index]) 
			# clone + mutate the clone
			clone = copy.deepcopy(self.agents[index])
			clone.mutate()
			new_agents.append(clone)

		# assign the new agents list
		self.agents = new_agents
		


	def evolution_step(self):
		survivors = self.split_population()
		self.create_clones(survivors)
		for i in range(len(self.scores)):
			self.scores[i] = 0





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
			agent_thread = threading.Thread(target = self.test_agents_thread, args = (i, inputs, solutions))
			threads.append(agent_thread)

		# Start all threads
		for single in threads:
			single.start()

		# Wait for all of them to finish
		for wait in threads:
			wait.join()

		self.best_score = max(self.scores)











































































