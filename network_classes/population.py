import random

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































































