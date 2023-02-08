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
		self.weight_scores = []
		
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
			self.weight_scores.append( 0 )


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

		self.time_weight_scoring = 0






#########################################################################################################################
#													Scoring Functions													#
#########################################################################################################################


	def reset_all_scores(self):
		for i in range(len(self.agents)):
			self.scores[i] 				= 0
			self.base_scores[i] 		= 0
			self.test[i]              	= []
			self.confidence_scores[i] 	= [0,0,0,0]
			self.weight_scores[i] = 0







	def tanh_confidence_scoring(self, agent_index, output, solution):
		# score values
		# -------------
		abs_output = abs(output)
		if abs_output >= 0.90:
			post_tanh = (1/2) * log((1+0.90)/(1-0.90)) 
		else:
			post_tanh = (1/2) * log((1+abs_output)/(1-abs_output)) 


		score = 1.80**post_tanh

		# assigning scores
		# -----------------
		# right
		if (solution > 0 and output > 0) or (solution <= 0 and output <= 0):
			self.scores[agent_index]      += score

		# wrong
		elif (solution > 0 and output <= 0) or (solution <= 0 and output > 0):
			self.scores[agent_index] -= score





	# weight scoring
	# ---------------
	def weight_scoring(self, agent_index):

		weight_score = 0

		# all layers besides second to last
		# ----------------------------------
		impact = 0.003
		for i in range(len(self.agents[agent_index].layers)):
			layer_weight = 0
			for neuron in self.agents[agent_index].layers[i]:
				for weight in neuron.weights:
					layer_weight += weight**2 #+= abs(weight)

			#weight_score += (layer_weight*impact)/len(self.agents[agent_index].layers[i])
			weight_score += (layer_weight*impact)
		
		# connections to final layer
		# ---------------------------
		#final_weight = 0
		#final_impact = 1.0
		#for neuron in self.agents[agent_index].layers[-2]:
			#for weight in neuron.weights:
				#final_weight += weight**2

		#weight_score += (final_weight*final_impact)/len(self.agents[agent_index].layers[-2])
		#weight_score += (final_weight*final_impact)


		# set agent weight score and return
		# ----------------------------------
		self.weight_scores[agent_index] = weight_score
		return weight_score







	# NEW STUFF
	# -------------------------------------------------

	def timed_test_agents_double(self, inputs, solutions, solution_variance):
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


					# grabbing all outputs form the first test
					# -----------------------------------------
					first_estimates = []
					for k in range(len(self.agents[i].layers[-1])):
						first_estimates.append(self.agents[i].layers[-1][k].value)

					team_1_points = self.agents[i].layers[-1][ 0].value
					team_2_points = self.agents[i].layers[-1][14].value

					
					# SECOND way around
					# ------------------
					set_inputs_time = time()
					self.agents[i].set_inputs(inputs[current+1])
					self.time_setting_inputs += (time()-set_inputs_time)

					second_test, calc_time, send_time = self.agents[i].calculate_value()
					self.time_calculating_values += calc_time
					self.time_sending_values     += send_time





					# Scoring
					# ------------------
					# final decision (for the final decision we will use the first way around)
					tanh_scoring_time = time()


					# percent errors
					# ---------------
					offset      = len(solutions[current+1])//2
					output      = []
					total_error = 0
					for k in range(len(self.agents[i].layers[-1])):
						if k < offset:
							first_value  = first_estimates[k]
							second_value = self.agents[i].layers[-1][k+offset].value
						else:
							first_value  = first_estimates[k]
							second_value = self.agents[i].layers[-1][k-offset].value

						# % error = (estimated-actual)/estimated
						estimated = (first_value-second_value)/2
						actual 	  = solutions[current][k]

						output.append( estimated )
						
						#error = abs(actual-estimated)
						error = (actual-estimated)**2
						error = error / (solution_variance[k]) # divide the error by its (variance*100)

						total_error += error

					# average error
					self.scores[i] -= ((total_error / len(solutions[current+1])) * 5)
					




					#      0,      1,      2,       3,       4,      5,      6,     7,     8,      9,    10,     11,     12,    13
					#  "Pts",  "FGM",  "FGA",  "FGM3",  "FGA3",  "FTM",  "FTA",  "OR",  "DR",  "Ast",  "TO",  "Stl",  "Blk",  "PF"
					#     14,     15,     16,      17,      18,     19,     20,    21,    22,     23,    24,     25,     26,    27
                	# "xPts", "xFGM", "xFGA", "xFGM3", "xFGA3", "xFTM", "xFTA", "xOR", "xDR", "xAst", "xTO", "xStl", "xBlk", "xPF"


					# correct overall score differentials
					# ------------------------------------
					team_1_points -= self.agents[i].layers[-1][14].value
					team_2_points -= self.agents[i].layers[-1][ 0].value

					if team_1_points + team_2_points > 0 and solutions[current][0] > solutions[current][14]:
						self.base_scores[i] += 1
						self.scores[i] += 1


					# correct fgm, 3fgm, and ftm in relation to attempted
					# ----------------------------------------------------
					#if output[]


					# points add up from made shots
					# ------------------------------
					points_estimate_score = 0

					estimated_points_1     = (2*output[ 1] + output[ 3] + output[ 5])
					normal_points_1        = output[ 0]
					points_estimate_score += (normal_points_1 - estimated_points_1)**2

					estimated_points_2     = (2*output[15] + output[17] + output[19])
					normal_points_2        = output[14]
					points_estimate_score += (normal_points_2 - estimated_points_2)**2


					self.scores[i]        -= (points_estimate_score/2)*3




					self.time_tanh_scoring += (time()-tanh_scoring_time)

					# iterate j an extra value since we use two per loop
					current += 2




				# weights
				# -------
				self.scores[i] -= self.weight_scoring(i)



				self.num_calculations += 1


				# Weight Decay
				# -------------
				#weight_scoring_time = time()
				#self.scores[i] -= self.weight_scoring(i)
				#self.time_weight_scoring += (time()-weight_scoring_time)


		self.time_threading += (time()-threading_time)










#########################################################################################################################
#													Testing Functions													#
#########################################################################################################################


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

					# SECOND way around
					self.agents[i].set_inputs(inputs[current+1])
					second_test, calc_time, send_time = self.agents[i].calculate_value()

					# final decision (for the final decision we will use the first way around)
					output = (first_test - second_test)

					# score the agents final evaluation
					self.tanh_confidence_scoring(i, output, solutions[current])
					self.confidence_scoring(i, output, solutions[current])
					self.logic_scoring(i, output, solutions[current])           # normal scoring

					# iterate j an extra value since we use two per loop
					current += 2


				# Weight Decay
				# -------------  
				self.scores[i] -= self.weight_scoring(i)

				








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



	def create_clones(self, survivor_indicis, clones_per=1):
		# create new list of agents
		new_agents = list(range( len(self.agents) ))

		# create clones and reassign originals
		clone_time = time()

		clones = []
		for index in survivor_indicis:
			new_agents[index] = self.agents[index].agent_copy()   # original

			clone = self.agents[index].agent_copy()  # clone
			clones.append( clone )	#clones.append(copy.deepcopy(self.agents[index])) # clone

		self.time_cloning += (time() - clone_time)
 
		# fill empty indicis with the clones
		for i in range(len(new_agents)):
			if type(new_agents[i]) == int:
				# reset the score of the clone
				self.confidence_scores[i] = [0,0,0,0] 
				self.base_scores[i]       = 0
				self.scores[i]            = 0 
				self.test[i] 			  = []
				self.weight_scores[i]       = 0
				
				# assign and mutate new clone
				new_agents[i] = clones.pop(0)

				mutation_time = time()
				new_agents[i].mutate()
				new_agents[i].mutate()
				self.time_mutating += (time()-mutation_time)


		# assign the new agents list
		self.agents = new_agents
		


	def evolution_step(self, reset_confidence_scores=True):
		survivors = self.split_population(survivor_percentage=0.5)
		self.create_clones(survivors, clones_per=1)
		









	























































