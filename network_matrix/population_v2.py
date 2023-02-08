'''

Speed Changes:
---------------
change the layer structure as well as the calculation algorithms slightly



Statistic Tracking Changes:
----------------------------
time
steps
accuracy
score








'''

import os

import numpy as np

from math import exp
from time import time



class Population:


	def __init__(self, num_inputs, num_outputs, learning_rate=0.01):

		# specifications
		# ---------------
		self.num_inputs = num_inputs
		self.l_outputs  = num_outputs
		self.l_neurons  = int(num_inputs // (3/2)) + self.l_outputs


		self.middle_layers = 3
		#self.w_range       = 0.25
		self.learning_rate = learning_rate


		self.current_batch_size = 0



		# statistics
		# -----------
		self.last_10_step_times = [1,1,1,1,1,1,1,1,1,1]






		# layers
		# -------
		self.layers = []

		# input layer
		self.layers.append( {"values": np.empty(self.num_inputs, np.float32)} )

		# all other layers
		prev_layer_length = self.num_inputs
		this_layer_length = self.l_neurons
		for i in range(self.middle_layers+1):
			if i == self.middle_layers: this_layer_length = self.l_outputs
			
			# weight range for each layer based on a formula
			# -----------------------------------------------
			# formula from online
			w_range = np.random.randn(this_layer_length, prev_layer_length) * np.sqrt(1/prev_layer_length)
			

			weights  = np.random.uniform(-w_range, w_range, size=(this_layer_length, prev_layer_length)).astype(np.float32, casting='unsafe', copy=True)
			biases   = np.zeros(this_layer_length, np.float32)
			values   = np.zeros(this_layer_length, np.float32)

			v_cost     = np.zeros(this_layer_length, np.float32)

			w_cost     = np.zeros((this_layer_length, prev_layer_length), np.float32)
			w_cost_sum = np.zeros((this_layer_length, prev_layer_length), np.float32)

			b_cost     = np.zeros(this_layer_length, np.float32)
			b_cost_sum = np.zeros(this_layer_length, np.float32)

			self.layers.append({"weights"    : weights,
								"biases"     : biases,
								"values"     : values,

								"v_cost"     : v_cost,

								"w_cost"     : w_cost,
								"w_cost_sum" : w_cost_sum,

								"b_cost"     : b_cost,
								"b_cost_sum" : b_cost_sum})

			prev_layer_length = this_layer_length










	# new version with matrix multiplecation
	def calculate_values(self):

		for i in range(1, len(self.layers)):
			weighted_values    = np.multiply(self.layers[i-1]["values"], self.layers[i]["weights"])
			summed_connections = np.sum(weighted_values, axis=1)
			with_biases        = np.sum([summed_connections, self.layers[i]["biases"]],  axis=0)

			if i < len(self.layers)-1: 	self.layers[i]["values"] = np.tanh(with_biases) # tanh activation function on each layer besides the final layer
			else: 						self.layers[i]["values"] = with_biases 			# final layer has no activation function

		# return values of final layer
		return np.copy(self.layers[-1]["values"])







	def calc_grad_descent(self, solutions, variance):
		# for each of the output neurons, cost deriv is 2(value-actual)
		# since i actually don't do tanh to the final layer, the cost function needs to be different for it

		# do last layer first
		a1_1 = np.divide(np.multiply(np.subtract(self.layers[-1]["values"], solutions), 2), variance)
		a1_2 = np.subtract(1, np.square(self.layers[-1]["values"]))

		# weight decay
		# -------------
		weight_L2 = np.multiply(np.average(np.square(self.layers[-1]["weights"]), axis=1), 0.0001)
		a1_1 = np.add(a1_1, weight_L2)


		#self.layers[-1]['b_cost'] = np.multiply(a1_1, a1_2)
		self.layers[-1]['b_cost'] = a1_1
		self.layers[-1]['w_cost'] = np.outer(self.layers[-1]['b_cost'], self.layers[-2]["values"])

		self.layers[-1]['v_cost'] = np.sum(self.layers[-1]["b_cost"][:, None] * self.layers[-1]["weights"], axis=0)

		self.layers[-1]["w_cost_sum"] = np.add(self.layers[-1]["w_cost"], self.layers[-1]["w_cost_sum"])
		self.layers[-1]["b_cost_sum"] = np.add(self.layers[-1]["b_cost"], self.layers[-1]["b_cost_sum"])


		for i in reversed(range(1, len(self.layers)-1)):

			a1_1 = self.layers[i+1]["v_cost"] 
			a1_2 = np.subtract(1, np.square(self.layers[i]["values"]))

			# weight decay
			# -------------
			weight_L2 = np.multiply(np.average(np.square(self.layers[i]["weights"]), axis=1), 0.0001)
			a1_1 = np.add(a1_1, weight_L2)


			self.layers[i]["b_cost"] = np.multiply(a1_1, a1_2)
			self.layers[i]['w_cost'] = np.outer(self.layers[i]["b_cost"], self.layers[i-1]["values"])

			self.layers[i]['v_cost'] = np.sum(self.layers[i]["b_cost"][:, None] * self.layers[i]["weights"], axis=0)


			self.layers[i]["w_cost_sum"] = np.add(self.layers[i]["w_cost"], self.layers[i]["w_cost_sum"])
			self.layers[i]["b_cost_sum"] = np.add(self.layers[i]["b_cost"], self.layers[i]["b_cost_sum"])
		
		self.current_batch_size += 1








	# use the previously calculated gradient descent to get the total
	def update_weights(self):
		# iterate through all weights and costs
		for i in reversed(range(1, len(self.layers))):
			# calculate the average cost and adjust weights
			weight_cost_average  = np.divide(self.layers[i]["w_cost_sum"], self.current_batch_size)
			weight_adjusted_rate = np.multiply(self.learning_rate, weight_cost_average)
			self.layers[i]["weights"] = np.subtract(self.layers[i]["weights"], weight_adjusted_rate)

			# calculate the average cost and adjust biases
			bias_cost_average  = np.divide(self.layers[i]["b_cost_sum"], self.current_batch_size)
			bias_adjusted_rate = np.multiply(self.learning_rate, bias_cost_average)
			self.layers[i]["biases"] = np.subtract(self.layers[i]["biases"], bias_adjusted_rate)

			# reset costs and batch size
			self.layers[i]["w_cost_sum"] = np.multiply(0, self.layers[i]["w_cost_sum"])
			self.layers[i]["b_cost_sum"] = np.multiply(0, self.layers[i]["b_cost_sum"])
		self.current_batch_size      = 0






	def train_and_test(self, data, steps):
		#"                                                                         steps           time      remaining  "
		#"                                                                     ----------  -------------  ------------- "
		# [################                                  ] -   32.5% |      0026/0080  01h 02m 35.4s  01h 02m 35.4s     ...
		print("Training Progress:                             |  cost     acc  |     steps           time      remaining  ")
		print("                                               | ------  ------ | ----------  -------------  ------------- ")


		# given data
		training_inputs 	= data[0]
		training_solutions 	= data[1]
		training_variance   = data[2]
		testing_inputs 		= data[3]
		testing_solutions 	= data[4]
		testing_variance    = data[5]

		# timing
		calculating_values_time = 0
		gradient_descent_time   = 0
		step_time = 0
		scoring_time = 0

		# tracking
		average_costs  = []
		all_accuracies = []

		for j in range(steps):
			start_step_time = time()

			total_cost = []
			accuracy   = 0
			for i in range(len(training_inputs)):
				# set inputs
				self.layers[0]["values"] = training_inputs[i]

				# calculate the values
				calc_time = time()
				output = self.calculate_values()
				calculating_values_time += (time()-calc_time)

				# update the gradient descent
				grad_time = time()
				self.calc_grad_descent(training_solutions[i], training_variance)
				gradient_descent_time += (time()-grad_time)

				# tracking its scores
				score_time = time()
				total_cost.append(np.sum(np.square(np.subtract(output, training_solutions[i]))))

				accuracy += self.check_accuracy(output, training_solutions[i])				

				scoring_time += (time()-score_time)


			# every x steps, update weights, test it with the testing data, output info
			average_costs.append(  sum(total_cost) / len(total_cost)      )
			all_accuracies.append(        accuracy / len(training_inputs) )
			self.update_weights()

			# running average steps
			self.last_10_step_times.pop(0)
			self.last_10_step_times.append((time()-start_step_time))

			step_time += (time()-start_step_time)


			# printing progress
			self.progress_bar(j, steps, step_time, average_costs[-1], all_accuracies[-1])


		return [steps, calculating_values_time, gradient_descent_time, scoring_time, step_time], average_costs, all_accuracies














			



	def check_accuracy(self, output, solutions):
		#  0: WFG%2       1: WFGA2       2: WFG%3       3: WFGA3       4: WFT%        5: WFTA      
		#  6: WOR         7: WDR         8: WAst        9: WTO        10: WStl       11: WBlk       12: WPF
		# 13: LFG%2      14: LFGA2      15: LFG%3      16: LFGA3      17: LFT%       18: LFTA
		# 19: LOR        20: LDR        21: LAst       22: LTO        23: LStl       24: LBlk       25: LPF 

		points_for =  2*output[ 0]*output[ 1] +  3*output[ 2]*output[ 3] +    output[ 4]*output[ 5]
		points_aga = -2*output[13]*output[14] + -3*output[15]*output[16] + -1*output[17]*output[18]

		sol_for =  2*solutions[ 0]*solutions[ 1] +  3*solutions[ 2]*solutions[ 3] +    solutions[ 4]*solutions[ 5]
		sol_aga = -2*solutions[13]*solutions[14] + -3*solutions[15]*solutions[16] + -1*solutions[17]*solutions[18]

		if points_for > 0 and points_aga < 0:
			if   abs(points_for) > abs(points_aga) and abs(sol_for) > abs(sol_aga): return 1
			elif abs(points_for) < abs(points_aga) and abs(sol_for) < abs(sol_aga): return 1

		return 0





	def report_training_progress(self, times):
		# timing stats
		# -------------
		cv  = round(  times[1], 2)
		cvp = round( (times[1]/times[4])*100, 2)
		cvs = round(  times[1]/times[0], 2)

		gd  = round(  times[2], 2)
		gdp = round( (times[2]/times[4])*100, 2)
		gds = round(  times[2]/times[0], 2)

		sc  = round(  times[3], 2)
		scp = round( (times[3]/times[4])*100, 2)
		scs = round(  times[3]/times[0], 2)

		step   = round(  times[4], 2)
		step_p = round( (times[4]/times[4])*100, 2)
		step_s = round(  times[4]/times[0], 2)


		# weight decay stats
		# -------------------
		L1_sum = 0
		L2_sum = 0
		num_connections = 0
		for i in range(1, len(self.layers)):
			num_connections += self.layers[i]["weights"].size
			L1_sum += np.sum( np.sum(self.layers[i]["weights"], axis=1) )
			L2_sum += np.sum( np.sum(np.square(self.layers[i]["weights"]), axis=1) )

		L1_avg = round(L1_sum/num_connections, 4)
		L2_avg = round(L2_sum/num_connections, 4)



		report = """ 
             time(s)   percent  per step       | steps:         {:>8}
            --------  --------  --------       | learning rate: {:>8}
calc value  {:>8}  {:>8}% {:>8}       | # connections: {:>8}
grad dscnt  {:>8}  {:>8}% {:>8}       | L1 (sum):      {:>8}
scoring     {:>8}  {:>8}% {:>8}       | L2 (**2):      {:>8}
step time   {:>8}  {:>8}% {:>8}
""".format( times[0], self.learning_rate,
			cv, cvp, cvs,    num_connections,
			gd, gdp, gds,    L1_avg,
			sc, scp, scs,    L2_avg,
			step, step_p, step_s)

		print(report)


	def time_display(self, seconds):
		output = ""

		if seconds > 60:
			minutes = int(seconds // 60)
			seconds = seconds  % 60

			if minutes > 60:
				hours   = minutes // 60
				minutes = minutes  % 60
				output = "{:>2}h {:>2}m {:>2}s".format(hours, minutes, round(seconds, 1))

			else:
				output = "{:>2}m {:>2}s".format(minutes, round(seconds, 1))

		else:
			output = "{:>2}s".format(round(seconds, 1))

		return output





	def progress_bar(self, current_step, total_steps, current_time, prev_cost, prev_acc):

		current_step = current_step+1
		
		current_pct  = current_step/total_steps
		bar_length   = int(33 * current_pct)
		blank_length = 33-bar_length

		bar   = "#" * bar_length
		blank = " " * blank_length


		progress  = round((current_step/total_steps)*100, 2)
		time_so_far    = round(   current_time, 1)
		#time_remaining = round( ((current_time*total_steps)/(current_step)-current_time) , 1)

		# running average time remaining
		running_average = (sum(self.last_10_step_times)/len(self.last_10_step_times))
		time_remaining  = round( (total_steps-current_step)*running_average, 1 ) 


		steps = "{}/{}".format(current_step, total_steps)


		output = "[{}{}] - {:>6}%  | {:>6}  {:>5}% | {:>10}  {:>13}  {:>13}     ".format(	bar, blank, 
																							progress, 
																							round(prev_cost, 3), round(prev_acc*100, 2),
																							steps,
																							self.time_display(time_so_far),
																							self.time_display(time_remaining))
		
		print(output, end="\r")





	def manual_round(self, x, decimals):
		power = 10.0**decimals
		return int(x*power)/power

	def box_score_preview(self, inputs, solutions, variances, game_id, maximums, headers_list):
		# calculations
		# -------------
		self.layers[0]["values"] = inputs[game_id]

		raw_outputs = self.calculate_values()

		outputs = np.multiply( raw_outputs, maximums)
		actual  = np.multiply( np.copy(solutions[game_id]), maximums)

		error = np.multiply(np.divide(np.subtract(raw_outputs, solutions[game_id]), raw_outputs), 100)
		cost  = np.square(np.subtract(raw_outputs,solutions[game_id]))
		var_cost = np.divide(cost, variances)


		points_for =  2*outputs[ 0]*outputs[ 1] +  3*outputs[ 2]*outputs[ 3] +    outputs[ 4]*outputs[ 5]
		points_aga = -2*outputs[13]*outputs[14] + -3*outputs[15]*outputs[16] + -1*outputs[17]*outputs[18]

		points_for_a =  2*actual[ 0]*actual[ 1] +  3*actual[ 2]*actual[ 3] +    actual[ 4]*actual[ 5]
		points_aga_a = -2*actual[13]*actual[14] + -3*actual[15]*actual[16] + -1*actual[17]*actual[18]


		total_cost = self.manual_round( np.sum(cost    ), 3 )
		total_varc = self.manual_round( np.sum(var_cost), 3 )
		avg_cost   = self.manual_round( total_cost/self.l_outputs, 3 )
		avg_varc   = self.manual_round( total_varc/self.l_outputs, 3 )




		# printing
		# ---------
		lines = {"1"       : "       ",
				 "actual1" : "actual ",
				 "model1"  : "model  ",
				 "2"       : "       ",
				 #"error1"  : "% err  ",
				 "cost1"   : "cost   ",
				 "var_c1"  : "var_c  ",
				 "3"       : "",
				 "4"       : "       ",
				 "actual2" : "actual ",
				 "model2"  : "model  ",
				 "5"       : "       ",
				 #"error2"  : "% err  ",
				 "cost2"   : "cost   ",
				 "var_c2"  : "var_c  "}

		percentages = [0,2,4, 13,15,17]
		for i in range(len(outputs)):
			if i in percentages: 	decimals = 3
			else: 					decimals = 1

			if i < len(outputs)//2:
				lines["1"      ] += "{:>6} ".format(headers_list [i])
				lines["actual1"] += "{:>6} ".format(self.manual_round(actual [i], decimals))
				lines["model1" ] += "{:>6} ".format(self.manual_round(outputs[i], decimals))
				lines["2"      ] += "{:>6} ".format("-----")
				#lines["error1" ] += "{:>6} ".format(self.manual_round(error  [i], 2))
				lines["cost1"  ] += "{:>6} ".format(self.manual_round(cost    [i], 2))
				lines["var_c1" ] += "{:>6} ".format(self.manual_round(var_cost[i], 2))
				
			else:
				lines["4"      ] += "{:>6} ".format(headers_list [i])
				lines["actual2"] += "{:>6} ".format(self.manual_round(actual [i], decimals))
				lines["model2" ] += "{:>6} ".format(self.manual_round(outputs[i], decimals))
				lines["5"      ] += "{:>6} ".format("-----")
				#lines["error2" ] += "{:>6} ".format(self.manual_round(error  [i], 2))
				lines["cost2"  ] += "{:>6} ".format(self.manual_round(cost    [i], 2))
				lines["var_c2" ] += "{:>6} ".format(self.manual_round(var_cost[i], 2))
				
		for line in lines.keys():
			print(lines[line])
		print()
		print("                model  actual                      total   average")
		print("               ------  ------                   --------  --------")
		print("team 1 score:  {:>6}  {:>6}            cost:  {:>8}  {:>8}".format(round(points_for, 1), round(points_for_a, 1), total_cost, avg_cost))
		print("team 2 score:  {:>6}  {:>6}            var_c: {:>8}  {:>8}".format(round(points_aga, 1), round(points_aga_a, 1), total_varc, avg_varc))
		print("               ------  ------")
		print("team 1 win?:   {:>6}  {:>6}".format( abs(points_for) > abs(points_aga), abs(points_for_a) > abs(points_aga_a) ))






	def print_magnitudes_report(self):
		decimals = 3
		# abs_average, abs_median, sum, max, min
		keys = ["values", "biases", "weights"]

		header = """
           abs_avg   abs_med       sum       max       min"""

		all_data = []

		for i in range(1, len(self.layers)):
			data = { "values":[], "biases":[], "weights":[] }

			for key in keys:
				data[key].append( self.manual_round(np.average(np.abs(self.layers[i][key])), decimals) )
				data[key].append( self.manual_round(np.median(np.abs(self.layers[i][key])), decimals) )
				data[key].append( self.manual_round(np.sum(self.layers[i][key]), decimals) )
				data[key].append( self.manual_round(np.max(self.layers[i][key]), decimals) )
				data[key].append( self.manual_round(np.min(self.layers[i][key]), decimals) )

			all_data.append(data)
			out = """
          --------  --------  --------  --------  --------
values    {:>8}  {:>8}  {:>8}  {:>8}  {:>8}
biases    {:>8}  {:>8}  {:>8}  {:>8}  {:>8}
weights   {:>8}  {:>8}  {:>8}  {:>8}  {:>8}""".format( 	data["values" ][0], data["values" ][1], data["values" ][2], data["values" ][3], data["values" ][4],
														data["biases" ][0], data["biases" ][1], data["biases" ][2], data["biases" ][3], data["biases" ][4],
														data["weights"][0], data["weights"][1], data["weights"][2], data["weights"][3], data["weights"][4])
			header += out
			

		out0 = """
           abs_avg   abs_med       sum       max       min
          --------  --------  --------  --------  --------
values    {:>8}  {:>8}  {:>8}  {:>8}  {:>8}
biases    {:>8}  {:>8}  {:>8}  {:>8}  {:>8}
weights   {:>8}  {:>8}  {:>8}  {:>8}  {:>8}
		""".format( data["values" ][0], data["values" ][1], data["values" ][2], data["values" ][3], data["values" ][4],
					data["biases" ][0], data["biases" ][1], data["biases" ][2], data["biases" ][3], data["biases" ][4],
					data["weights"][0], data["weights"][1], data["weights"][2], data["weights"][3], data["weights"][4]
			)


		print(header)





































































































































































































