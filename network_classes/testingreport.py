

# imports
from time import time


# custom modules
from .neuron 		import Neuron
from .agent  		import Agent
from .population 	import Population


import sys
sys.path.append("..")
from extra.progress_package import *


class TestingReport:
	'''


	'''
	def __init__(self, population):

		# Population being observed
		# --------------------------
		self.population = population


		# current values
		# ---------------
		self.current_split_timer = 0
		self.current_split_steps = 0

		self.current_split_steps_sec = 0
		self.current_split_secs_step = 0

		self.current_split_evolution_timer = 0
		self.current_split_scoring_timer   = 0

		self.current_split_max_conf  = 0
		self.current_split_max_base  = 0
		self.current_split_accuracy  = 0


		# total values
		# -------------
		self.total_timer = 0
		self.total_steps = 0

		self.total_steps_sec = 0
		self.total_secs_step = 0

		self.total_evolution_timer = 0
		self.total_scoring_timer   = 0

		self.total_max_conf  = 0
		self.total_max_base  = 0
		self.total_accuracy  = 0


		# previous values
		# ----------------
		self.prev_timer     = 0
		self.prev_steps     = 0

		self.prev_steps_sec = 0
		self.prev_secs_step = 0

		self.prev_evolution_timer = 0
		self.prev_scoring_timer   = 0

		self.prev_max_conf  = 0
		self.prev_max_base  = 0
		self.prev_accuracy  = 0


		# score lists
		# ------------
		self.base_scores 		= self.population.base_scores
		self.conf_scores 		= self.population.scores
		self.conf_scores_list 	= self.population.confidence_scores



		# try storing them all in a list
		# -------------------------------
		#     0,    1,          2,      3
		# total, prev, this_split, change

		#self.timer 				= [0,0,0,0]
		#self.steps 				= [0,0,0,0]

		#self.steps_sec 			= [0,0,0,0]
		#self.secs_step 			= [0,0,0,0]

		#self.evolution_timer 		= [0,0,0,0]
		#self.scoring_timer   		= [0,0,0,0]

		#self.max_conf  			= [0,0,0,0]
		#self.max_base  			= [0,0,0,0]
		#self.accuracy  			= [0,0,0,0]










	def update_scores(self):
		# update scores
		# -------------
		self.base_scores 		= self.population.base_scores
		self.conf_scores 		= self.population.scores
		self.conf_scores_list 	= self.population.confidence_scores


		# previous values
		# ----------------
		self.prev_max_conf  = self.total_max_conf
		self.prev_max_base  = self.total_max_base
		self.prev_accuracy  = self.total_accuracy


		# total values
		# -------------
		self.total_max_conf  = max(self.conf_scores )
		self.total_max_base  = max(self.base_scores)

		# current values
		# ---------------
		self.current_split_max_conf  = 0
		self.current_split_max_base  = 0
		self.current_split_accuracy  = 0








#########################################################################################################################
#										printing detailed reports														#
#########################################################################################################################


	def print_report(self, update_data=False):
		'''
		Prints a report on the population using the currently stored statastics.

		'''
		if update_data:
			self.update_scores()


		timer_l = [ round(self.total_timer,2),           round(self.prev_timer,2),           round(self.current_split_timer,2)]
		steps_l = [ self.total_steps,                    self.prev_steps,                    self.current_split_steps]
		steps_s = [ self.total_steps_sec,                self.prev_steps_sec,                self.current_split_steps_sec]
		s_steps = [ self.total_secs_step,                self.prev_secs_step,                self.current_split_secs_step]
		conf_l  = [ round(self.total_max_conf),          round(self.prev_max_conf),          round(self.current_split_max_conf - self.prev_max_conf)]
		base_l  = [ self.total_max_base, 			     self.prev_max_base,                 self.current_split_max_base       - self.prev_max_base]
		acc_l   = [ self.total_accuracy,                 self.prev_accuracy,                 self.current_split_accuracy       - self.prev_accuracy] 
		evo_t   = [ round(self.total_evolution_timer,2), round(self.prev_evolution_timer,2), round(self.current_split_evolution_timer,2)]
		score_t = [ round(self.total_scoring_timer,2),   round(self.prev_scoring_timer,2),   round(self.current_split_scoring_timer,2)]

		out = """

            current    prev       change  
            --------   --------   --------
timer      {:-8.2f}s  {:-8.2f}s  {:-8.2f}s
steps      {:-8}   {:-8}   {:-8} 
steps/sec  {:-8.2f}   {:-8.2f}   {:-8.2f} 
secs/step  {:-8.2f}s  {:-8.2f}s  {:-8.2f}s
evolu time {:-8.2f}s  {:-8.2f}s  {:-8.2f}s
score time {:-8.2f}s  {:-8.2f}s  {:-8.2f}s
            --------   --------   --------
conf score {:-8}   {:-8}   {:-8} 
base score {:-8}   {:-8}   {:-8} 
accuracy   {:-8.2f}%  {:-8.2f}%  {:-8.2f}%

		""".format(
			timer_l[0], timer_l[1], timer_l[2],
			steps_l[0], steps_l[1], steps_l[2],
			steps_s[0], steps_s[1], steps_s[2],
			s_steps[0], s_steps[1], s_steps[2],
			evo_t  [0], evo_t  [1], evo_t  [2],
			score_t[0], score_t[1], score_t[2],
			conf_l [0], conf_l [1], conf_l [2],
			base_l [0], base_l [1], base_l [2],
			acc_l  [0], acc_l  [1], acc_l  [2]
			)

		print(out)


		# print out final scores
		# -----------------------
		print("id - [ c r, nc r, nc w,  c w] | ovrl |  conf  ({})".format( round(self.current_split_max_conf) ))
		print("---------------------------------------------")
		#       0 - [2030,  209,  202, 1295] | 2974 | 2239

		for i in range(len(self.base_scores)):
		    # get the confidence scores
		    conf = self.conf_scores_list[i]
		    conf_scores_string = "[{:>4}, {:>4}, {:>4}, {:>4}]".format(conf[0], conf[1], conf[2], conf[3])
		    
		    print( "{:>2} - {} | {:>4} | {:>5}".format(i, conf_scores_string, self.base_scores[i], round(self.conf_scores[i])) )






	# utility for reports that only look at 1 agent
	def get_top_agent_index(self):
		max_score = 0
		max_index = 0
		for i in range(len(self.population.agents)):
		    if self.population.scores[i] > max_score:
		        max_score = self.population.scores[i]
		        max_index = i

		return max_index



	def print_top_agent(self):
		self.population.agents[ self.get_top_agent_index() ].print2()



	def size_report(self, all_attributes=True):
		print("{} steps".format(self.total_steps))
		self.population.agents[ self.get_top_agent_index() ].size_report(all_attributes=all_attributes)



	def evolution_time_report(self):
		t  = round(self.total_evolution_timer, 2)

		c  = round(self.population.time_cloning, 2)
		cp = round( (c/t)*100 ,2)

		m  = round(self.population.time_mutating, 2)
		mp = round( (m/t)*100 ,2)

		s   = round(self.population.time_splitting_population, 2)
		sp  = round( (s/t)*100 ,2)

		sd  = round(self.population.time_creating_score_dict, 2)
		sdp = round( (sd/t)*100 ,2)

		ss  = round(self.population.time_selecting_survivors, 2)
		ssp = round( (ss/t)*100 ,2)
	        
		print("""
            time     % total
            -------  -------
total       {0:>6}s
cloning     {1:>6}s  {2:>6}%
mutating    {3:>6}s  {4:>6}%
splitting   {5:>6}s  {6:>6}%
score dict  {7:>6}s  {8:>6}%
selecting   {9:>6}s  {10:>6}%
""".format(t, 
           c,cp, 
           m,mp,
           s,sp, 
           sd,sdp,
           ss,ssp))



	def scoring_time_report(self):
		total_scoring_time = round(self.total_scoring_timer, 2)

		#divided_by_time = self.population.num_calculations
		divided_by_time = 1


		calc_time   = round(self.population.time_calculating_values / divided_by_time, 2 )
		calc_time_p = round( (calc_time/total_scoring_time)*100, 2 )

		send_time   = round(self.population.time_sending_values / divided_by_time, 2 )
		send_time_p = round( (send_time/total_scoring_time)*100, 2 )

		set_time   = round(self.population.time_setting_inputs / divided_by_time, 2 )
		set_time_p = round( (set_time/total_scoring_time)*100, 2 )

		tanh_time   = round(self.population.time_tanh_scoring / divided_by_time, 2 )
		tanh_time_p = round( (tanh_time/total_scoring_time)*100, 2 )


		accounted_for   = round(sum([calc_time, send_time, set_time, tanh_time]), 2)
		accounted_for_p = round( (accounted_for/total_scoring_time)*100, 2 )



		# threading stuff
		# ---------------

		thread_time     = round(self.population.time_threading, 2 )
		thread_time_p   = round( (thread_time/total_scoring_time)*100, 2 )


		creating     = round(self.population.time_creating_threads, 2 )
		creating_p   = round( (creating/thread_time)*100, 2 )

		starting     = round(self.population.time_starting_threads, 2 )
		starting_p   = round( (starting/thread_time)*100, 2 )

		joining     = round(self.population.time_joining_threads, 2 )
		joining_p   = round( (joining/thread_time)*100, 2 )



		print("""
            time     % total
            -------  -------
total       {0:>6}s
calc val    {1:>6}s  {2:>6}%
send time   {3:>6}s  {4:>6}%
set time    {5:>6}s  {6:>6}%
inv tanh    {7:>6}s  {8:>6}%
            -------  -------
acc for     {9:>6}s  {10:>6}%
            -------  -------
thread time {11:>6}s  {12:>6}%
creating    {13:>6}s  {14:>6}%
starting    {15:>6}s  {16:>6}%
joining     {17:>6}s  {18:>6}%
""".format(total_scoring_time, 
           calc_time,  		calc_time_p, 
           send_time,  		send_time_p,
           set_time,      	set_time_p,
           tanh_time, 		tanh_time_p,
           accounted_for, 	accounted_for_p,

           thread_time, 	thread_time_p, 
           creating, 		creating_p, 
           starting, 		starting_p, 
           joining, 		joining_p
           ))








#########################################################################################################################
#												Testing Function														#
#########################################################################################################################


	def run_test(self, training_time, data, max_steps=1000):
		self.population.reset_all_scores()

		# set up data
		# ------------
		training_data      	= data[0]
		training_solutions 	= data[1]
		testing_data 		= data[2]
		testing_solutions 	= data[3]


		# for stat tracking
		# ------------------
		timer = 0
		steps = 0

		evolution_timer = 0
		scoring_timer   = 0


		# stats for loop control
		# -----------------------
		start_time = time()
		elapsed    = 0


		# loop start
		# -----------
		for i in range(max_steps):


			# check if loop should be done
		    # -----------------------------
		    elapsed = (time()-start_time)
		    if elapsed > training_time: 
		        progress_bar(training_time, training_time)
		        break
		    else: 
		        # show progress update
		        progress_bar(elapsed, training_time)

		    
		    # scoring
		    # --------
		    scoring_start = time()
		    #self.population.test_agents_threading_double(training_data, training_solutions) # (inputs_splits[i%10], solutions_splits[i%10])
		    self.population.timed_test_agents_double(training_data, training_solutions)
		    scoring_timer += (time() - scoring_start)


		    # evolution
		    # ----------
		    evolution_start = time()
		    self.population.evolution_step()
		    evolution_timer += (time() - evolution_start)
		    steps += 1


		# loop is over
		# -------------
		timer += (time() - start_time)
		    

		# check scores
		# -------------
		self.population.reset_all_scores()
		self.population.test_agents_double(testing_data, testing_solutions)


		# update all values
		# ------------------
		# score lists
		self.base_scores 		= self.population.base_scores
		self.conf_scores 		= self.population.scores
		self.conf_scores_list 	= self.population.confidence_scores

		# prev values
		self.set_prev()

		# current values
		self.current_split_timer = timer
		self.current_split_steps = steps

		self.current_split_steps_sec = round(steps/timer, 2)
		self.current_split_secs_step = round(timer/steps, 2)

		self.current_split_evolution_timer = evolution_timer
		self.current_split_scoring_timer   = scoring_timer

		self.current_split_max_conf  = max(self.conf_scores)
		self.current_split_max_base  = max(self.base_scores)
		self.current_split_accuracy  = round( ( self.current_split_max_base/(len(testing_solutions)//2))*100, 2 )

		# total values
		self.set_total()




	# helpers for tracking stats during testing function
	# ---------------------------------------------------
	def set_prev(self):

		# previous values
		# ----------------
		self.prev_timer     = self.total_timer
		self.prev_steps     = self.total_steps

		self.prev_steps_sec = self.total_steps_sec
		self.prev_secs_step = self.total_secs_step

		self.prev_evolution_timer = self.total_evolution_timer
		self.prev_scoring_timer   = self.total_scoring_timer

		self.prev_max_conf  = self.total_max_conf
		self.prev_max_base  = self.total_max_base 
		self.prev_accuracy  = self.total_accuracy


	def set_total(self):

		# total values
		# -------------
		self.total_timer += self.current_split_timer
		self.total_steps += self.current_split_steps

		self.total_steps_sec = self.current_split_steps_sec
		self.total_secs_step = self.current_split_secs_step

		self.total_evolution_timer += self.current_split_evolution_timer
		self.total_scoring_timer   += self.current_split_scoring_timer

		self.total_max_conf  = self.current_split_max_conf
		self.total_max_base  = self.current_split_max_base
		self.total_accuracy  = self.current_split_accuracy 











'''

[##################################################] - 100.0%                                       


            current    prev       change  
            --------   --------   --------
timer       6958.38s   5745.27s   1213.11s
steps           638        550         88 
secs/step     10.91s     10.45s     13.79s
evolu time  3450.85s   2752.17s    698.68s
score time  3507.39s   2992.98s    514.41s
            --------   --------   --------
conf score     2213       2213          0 
base score     2465       2465          0 
accuracy      65.98%     65.98%      0.00%

id - [ c r, nc r, nc w,  c w] | ovrl |  conf  (2213)
---------------------------------------------
 0 - [2059,  328,  340, 1009] | 2387 |  2020
 1 - [2370,   59,   63, 1244] | 2429 |  2154
 2 - [1756,  655,  525,  800] | 2411 |  2042
 3 - [2037,  338,  355, 1006] | 2375 |  2075
 4 - [1046,  873,  550, 1267] | 1919 |  -905
 5 - [2317,  100,   86, 1233] | 2417 |  2128
 6 - [2172,  282,  251, 1031] | 2454 |  2213
 7 - [1478,  987,  575,  696] | 2465 |  1997
 8 - [2203,  228,  182, 1123] | 2431 |  2193
 9 - [2264,  124,  124, 1224] | 2388 |  2031
10 - [   0, 2011, 1725,    0] | 2011 |   280
11 - [ 989,  868,  756, 1123] | 1857 |  -616
12 - [ 625, 1304, 1154,  653] | 1929 |  -108
13 - [1199,  390,  611, 1536] | 1589 | -1587
14 - [2360,  100,   90, 1186] | 2460 |  2030
15 - [   0, 1806, 1930,    0] | 1806 |  -205
16 - [1676,  122,  108, 1830] | 1798 | -1376
17 - [   0, 1907, 1829,    0] | 1907 |    52
18 - [   0, 1860, 1876,    0] | 1860 |   -18
19 - [1453,  608,  548, 1127] | 2061 |   380


'''


















# end