# embeddings_data_prep.py

import pandas as pd
import numpy as np

from copy import deepcopy
from statistics import stdev



##############################################################################################################################
# Initial Data Prep
##############################################################################################################################


def get_seasons(reg_season_df, seasons):

	recent_season_df = reg_season_df[reg_season_df.Season == seasons[0]]

	for i in range(1, len(seasons)):
		season = seasons[i]

		recent_season_df = pd.concat([recent_season_df, reg_season_df[reg_season_df.Season == (season)]], ignore_index = True)
		recent_season_df.reset_index()

	return recent_season_df



def df_to_list_of_columns(df, print_report=False):

	# Creates list of the column names from the df
	columns = df.columns.tolist()

	# Creates list of the columns
	df_columns = []
	for expected_column in columns:
		df_columns.append( df[expected_column].tolist() )


	# print a report and then return
	# -------------------------------
	if print_report:
		print("rows:", len(df_columns[0]), "  columns:", len(df_columns))


	return df_columns



def combine_season_and_team_ids(df_columns, season_index, team_id_indicis):
	updated_df_columns = deepcopy(df_columns)

	for i in range(len(df_columns[season_index])):
		season = df_columns[season_index][i]

		for j in team_id_indicis:
			old_id = df_columns[j][i]
			new_id = "{}_{}".format(old_id, season)

			updated_df_columns[j][i] = new_id


	return updated_df_columns







##############################################################################################################################
# Create Ranks Dictionary
##############################################################################################################################


def create_ranks_dictionary(recent_ranks_df_columns, print_report=False):

	ranks_dict = {}

	# 0: Season, 1: RankingDayNum, 2: SystemName, 3: TeamID, 4: OrdinalRank
	for i in range(len(recent_ranks_df_columns[0])):
		TeamID        = recent_ranks_df_columns[3][i]
		SystemName    = recent_ranks_df_columns[2][i]
		RankingDayNum = recent_ranks_df_columns[1][i]
		OrdinalRank   = recent_ranks_df_columns[4][i]
		
		# new team entry
		if TeamID not in ranks_dict:
			#                            day_num: [running average, [list of ranks]]
			ranks_dict[TeamID] = { RankingDayNum: [OrdinalRank, [[SystemName, OrdinalRank]]] }
			
		# existing team entry
		else:
			# new day entry
			if RankingDayNum not in ranks_dict[TeamID]:
				ranks_dict[TeamID][RankingDayNum] = [OrdinalRank, [[SystemName, OrdinalRank]]]
			
			# existing day entry
			else:
				current_sum = ranks_dict[TeamID][RankingDayNum][0] * len(ranks_dict[TeamID][RankingDayNum][1])
				new_sum     = current_sum + OrdinalRank
				new_average = new_sum / (len(ranks_dict[TeamID][RankingDayNum][1])+1)
				
				ranks_dict[TeamID][RankingDayNum][0] = new_average
				ranks_dict[TeamID][RankingDayNum][1].append([SystemName, OrdinalRank])



	# print out report and return the ranks dict
	# -------------------------------------------
	if print_report:
		example_ID  = list(ranks_dict.keys())[0]
		example_day = list(ranks_dict[example_ID].keys())[0]

		print(example_ID)
		print(ranks_dict[example_ID][example_day])


	return ranks_dict




def add_team_ranks_to_data(recent_season_df_columns, ranks_dict, print_report=False):
	original_length = len(recent_season_df_columns)

	# add columns for WTeam Rank and LTeam Rank
	recent_season_df_columns.insert(21, []) # 21
	recent_season_df_columns.append([])     # 35


	num_rows = len(recent_season_df_columns[0])

	for i in range(num_rows):
		DayNum = recent_season_df_columns[1][i]
		WTeam  = recent_season_df_columns[2][i]
		LTeam  = recent_season_df_columns[4][i]
		
		# Look up the most recent ranking for WTeam and LTeam
		
		# WTeam
		# ------
		WTeam_ranks = list(ranks_dict[WTeam].keys())
		WTeam_ranks.sort()
		
		recent_W_key = WTeam_ranks[0]
		for key in WTeam_ranks:
			# once the key we are on is bigger than the day of the game, take the previous key
			if key > recent_W_key:
				# add the WTeam Rank to the dict
				recent_season_df_columns[21].append( ranks_dict[WTeam][recent_W_key][0] )
				break
				
		# LTeam
		# ------
		LTeam_ranks = list(ranks_dict[LTeam].keys())
		LTeam_ranks.sort()
		
		recent_L_key = LTeam_ranks[0]
		for key in LTeam_ranks:
			# once the key we are on is bigger than the day of the game, take the previous key
			if key > recent_L_key:
				# add the LTeam Rank to the dict
				recent_season_df_columns[35].append( ranks_dict[LTeam][recent_L_key][0] )
				break
	
	# print report and return?
	# -------------------------
	if print_report:
		cols_report = "original columns: {} | after adding ranks: {}".format(original_length, len(recent_season_df_columns))
		print(cols_report)

		for i in range(len(recent_season_df_columns)):
			print( "{:>3} - {:>5}".format(i, len(recent_season_df_columns[i])))

















##############################################################################################################################
# Data Manipulation
##############################################################################################################################


def separate_fga_and_fgp(recent_season_df_columns, print_report=False):

	# get just the games from the specified season
	# ---------------------------------------------
	#given_df = df_columns

	# change df to list of columns
	# -----------------------------
	#columns = given_df.columns.tolist()

	# Creates list of the columns
	#orig_columns = []
	#for expected_column in columns:
		#temp = given_df[expected_column].tolist()
		#orig_columns.append(temp)

	orig_columns = deepcopy(recent_season_df_columns)

	#  0: Season      1: DayNum      2: WTeamID     3: WScore      4: LTeamID     5: LScore      6: WLoc      
	#  7: NumOT       8: WFGM        9: WFGA       10: WFGM3      11: WFGA3      12: WFTM       13: WFTA      
	# 14: WOR        15: WDR        16: WAst       17: WTO        18: WStl       19: WBlk       20: WPF       
	# 21: LFGM       22: LFGA       23: LFGM3      24: LFGA3      25: LFTM       26: LFTA       27: LOR       
	# 28: LDR        29: LAst       30: LTO        31: LStl       32: LBlk       33: LPF 


	# cut down to just the relevent statistic columns
	# ------------------------------------------------
	number_columns = [deepcopy(orig_columns[3])] + deepcopy(orig_columns[ 8:21]) + [deepcopy(orig_columns[5])] + deepcopy(orig_columns[21:])

	#  0: WScore      1: WFGM        2: WFGA        3: WFGM3       4: WFGA3       5: WFTM        6: WFTA      
	#  7: WOR         8: WDR         9: WAst       10: WTO        11: WStl       12: WBlk       13: WPF
	# 14: LScore     15: LFGM       16: LFGA       17: LFGM3      18: LFGA3      19: LFTM       20: LFTA
	# 21: LOR        22: LDR        23: LAst       24: LTO        25: LStl       26: LBlk       27: LPF 


	# customize the stats to my new preferences
	# ------------------------------------------

	# change FGM and FGA to not include FGM3 and FGA3
	for i in range(len(number_columns[1])):
		number_columns[ 1][i] = number_columns[ 1][i]-number_columns[ 3][i] #  1: WFGM -  3: WFGM3
		number_columns[ 2][i] = number_columns[ 2][i]-number_columns[ 4][i] #  2: WFGA -  4: WFGA3

		number_columns[15][i] = number_columns[15][i]-number_columns[17][i] # 15: LFGM - 17: LFGM3
		number_columns[16][i] = number_columns[16][i]-number_columns[18][i] # 16: LFGA - 18: LFGA3

	# change from FGM to FG%
	for i in range(len(number_columns[1])):
		# WTeam
		# ------
		number_columns[ 1][i] = number_columns[ 1][i]/number_columns[ 2][i] #  1: WFGM2 /  2: WFGA2
		
		if number_columns[ 4][i] == 0: number_columns[ 3][i] = 0.25
		else: 							number_columns[ 3][i] = number_columns[ 3][i]/number_columns[ 4][i] #  3: WFGM3 /  4: WFGA3

		if number_columns[ 6][i] == 0:  number_columns[ 5][i] = 0.70
		else: 							number_columns[ 5][i] = number_columns[ 5][i]/number_columns[ 6][i] #  5: WFTM  /  6: WFTA


		# LTeam
		# ------
		number_columns[15][i] = number_columns[15][i]/number_columns[16][i] # 15: LFGM2 / 16: LFGA2

		if number_columns[ 18][i] == 0: number_columns[17][i] = 0.25
		else: 							number_columns[17][i] = number_columns[17][i]/number_columns[18][i] # 17: LFGM3 / 18: LFGA3

		if number_columns[20][i] == 0: 	number_columns[19][i] = 0.70
		else: 							number_columns[19][i] = number_columns[19][i]/number_columns[20][i] # 19: LFTM  / 20: LFTA


	# get rid of points
	number_columns.pop(14) # 14: LScore
	number_columns.pop( 0) #  0: WScore


	#  0: WFG%2       1: WFGA2       2: WFG%3       3: WFGA3       4: WFT%        5: WFTA      
	#  6: WOR         7: WDR         8: WAst        9: WTO        10: WStl       11: WBlk       12: WPF
	# 13: LFG%2      14: LFGA2      15: LFG%3      16: LFGA3      17: LFT%       18: LFTA
	# 19: LOR        20: LDR        21: LAst       22: LTO        23: LStl       24: LBlk       25: LPF 

	headers_list = [ "FG%2",  "FGA",  "FG%3",  "FGA3",  "FT%",  "FTA",  "OR",  "DR",  "Ast",  "TO",  "Stl",  "Blk",  "PF", 
					"xFG%2", "xFGA", "xFG%3", "xFGA3", "xFT%", "xFTA", "xOR", "xDR", "xAst", "xTO", "xStl", "xBlk", "xPF"]

	# add back team IDSs
	team_IDs = []
	team_IDs.append(orig_columns[2]) # WTeam ID
	team_IDs.append(orig_columns[4]) # LTeam ID



	# print report
	# ------------------------
	if print_report:
		print("number columns   | rows: {}, columns: {}".format(len(number_columns),            len(number_columns[0])))
		print("original columns | rows: {}, columns: {}".format(len(recent_season_df_columns),  len(recent_season_df_columns[0])))



	# put all columns back (dont keep just the number columns)
	# ---------------------------------------------------------
	#number_columns
	col_ids = [ 0, 1,  2, 3,  4, 5,
			   13,14, 15,16, 17,18]

	orig_ids = [ 8, 9, 10,11, 12,13,
				21,22, 23,24, 25,26]

	current = 0
	fgp_columns = []
	for i in range(len(recent_season_df_columns)):
		if i in orig_ids:
			fgp_columns.append( number_columns[col_ids[current]].copy() )
			current += 1
		else:
			fgp_columns.append( recent_season_df_columns[i].copy() )


	# print report and return
	# ------------------------
	if print_report:
		print("fgp columns      | rows: {}, columns: {}".format(len(fgp_columns), len(fgp_columns[0])))


	return fgp_columns





def normalizing_stats(df_columns, mens=True):

	# Normalizing each input stat to 0-1
	# -----------------------------------

	# create columns of all actual data
	if mens:
		stats_columns = [df_columns[3].copy()] + deepcopy(df_columns[  8:22 ])
		additional    = [df_columns[5].copy()] + deepcopy(df_columns[ 22:   ])
	else:
		stats_columns = [df_columns[3].copy()] + deepcopy(df_columns[  8:21 ])
		additional    = [df_columns[5].copy()] + deepcopy(df_columns[ 21:   ])



	for i in range(len(stats_columns)):
		stats_columns[i] += additional[i]


	# getting averages/standard devation
	column_averages = []
	column_std = []
	for column in stats_columns:
		column_std.append(stdev(column))
		column_averages.append( sum(column)/len(column) )


	# get valid max candidates
	within_range = []
	for i in range(len(stats_columns)):
		within_range_column = []
		for value in stats_columns[i]:
			if value < (column_averages[i] + (column_std[i]*2.0)) and value > (column_averages[i] - (column_std[i]*2.0)):
				within_range_column.append(value)
		within_range.append(within_range_column)


	# now create max columns list for use
	max_columns = []
	for i in range(len(within_range)):
		max_columns.append( max(within_range[i]) )
	max_columns += max_columns


	# normalize all values
	if mens:
		normalized_indicis = [3,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
							  5, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
		reverse_indicis = [21, 35]
	else:
		normalized_indicis = [3,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
							  5, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
		reverse_indicis = []


	for i in range(len(normalized_indicis)):
		column_max  = max_columns[i]
		column      = normalized_indicis[i]
		for j in range(len( df_columns[column] )):
			if i in reverse_indicis:
				df_columns[column][j] = 1 - (df_columns[column][j]/column_max)
			else:
				df_columns[column][j] = (df_columns[column][j]/column_max)


	# create list of normalized variance values
	variance = []
	for i in range(len(column_std)):
		variance.append( (column_std[i]/max_columns[i])**2 )
	variance += variance


	# return max columns and variance lists
	# --------------------------------------
	output_max_columns  = max_columns.copy()
	output_variance     = variance.copy()


	return df_columns, output_max_columns, output_variance
























##############################################################################################################################
# Creating Rows for Inputs and Testing
##############################################################################################################################


# create input rows
# ------------------
def create_input_rows(W_data, L_data, womens=False):
	'''
	if womens, then there will be no strength value in the data
	
		0,      1,      2,      3,      4,     5,     6,    7,
	"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FT%", "FTA", "OR",
	   8,     9,   10,    11,    12,   13,    14,
	"DR", "Ast", "TO", "Stl", "Blk", "PF", "Str", 
	
	
	'''

	# 4 parts
	W_for = []
	W_aga = []
	L_for = []
	L_aga = []
	for j in range(len(W_data[1][0])):

		W_for_val = 0
		W_aga_val = 0
		W_games = W_data[0]
		for k in range(W_games):
			W_for_val += W_data[1][k][j]
			W_aga_val += W_data[2][k][j]
		W_for.append(W_for_val/W_games)
		W_aga.append(W_aga_val/W_games)


		L_for_val = 0
		L_aga_val = 0
		L_games = L_data[0]
		for k in range(L_games):
			L_for_val += L_data[1][k][j]
			L_aga_val += L_data[2][k][j]
		L_for.append(L_for_val/L_games)
		L_aga.append(L_aga_val/L_games)

	W_win_p = (sum(W_data[3]) / len(W_data[3]))
	L_win_p = (sum(L_data[3]) / len(L_data[3]))
	
	
	# additional data
	# ----------------
	W_4 = []
	W_5 = []
	L_4 = []
	L_5 = []
	for j in range(len(W_data[4][0])):
		W_4_val = 0
		W_5_val = 0
		W_games = len(W_data[4])
		for k in range(W_games):
			W_4_val += W_data[4][k][j]
			W_5_val += W_data[5][k][j]
		W_4.append(W_4_val/W_games)
		W_5.append(W_5_val/W_games)


		L_4_val = 0
		L_5_val = 0
		L_games = len(L_data[4])
		for k in range(L_games):
			L_4_val += L_data[4][k][j]
			L_5_val += L_data[5][k][j]
		L_4.append(L_4_val/L_games)
		L_5.append(L_5_val/L_games)
	
	
	# cut unnecessary values
	# -----------------------
	#  5 = ft%
	# 14 = strength (if mens)
	to_cut = [W_4, W_5, L_4, L_5]
	for values in to_cut:
		if not womens:
			values.pop(14)
		values.pop(5)

	to_cut = [W_aga, L_aga]
	for values in to_cut:
		values.pop(5)


	# compare some rows
	# ------------------
	# 5 is bad, 4 is good
	W_together = np.add( np.array(W_4), np.array(L_5) )
	L_together = np.add( np.array(L_4), np.array(W_5) )
	# separate the strength value and add the others together
	if not womens:
		W_aga_str = W_aga.pop(-1)
		L_aga_str = L_aga.pop(-1)

		W_extra = [W_win_p, L_aga_str]
		L_extra = [L_win_p, W_aga_str]

		#W_together = np.add(W_together, np.array(L_aga))
		#L_together = np.add(L_together, np.array(W_aga))

	else:
		W_extra = [W_win_p]
		L_extra = [L_win_p]
		#W_together = np.add(W_together, np.array(L_aga))
		#L_together = np.add(L_together, np.array(W_aga))

	
	# create input rows from data
	# ----------------------------
	new_input_row_1 = W_for.copy() + L_aga.copy() + list(W_together) + W_extra
	new_input_row_2 = L_for.copy() + W_aga.copy() + list(L_together) + L_extra
	
	#new_input_row_1 = W_for.copy() + list(W_together) + W_extra
	#new_input_row_2 = L_for.copy() + list(L_together) + L_extra
	#new_input_row_1 = W_for.copy() + W_4 + W_5 + [W_win_p] + L_aga.copy() 
	#new_input_row_2 = L_for.copy() + L_4 + L_5 + [L_win_p] + W_aga.copy()
	#new_input_row_1 = W_for.copy() + [W_win_p] + L_aga.copy() 
	#new_input_row_2 = L_for.copy() + [L_win_p] + W_aga.copy()
	

	orig_length = len(new_input_row_1)
	for j in range(len(new_input_row_1)):
		new_input_row_1.append(-new_input_row_2[j])
		new_input_row_2.append(-new_input_row_1[j])



	# not sure what this does tbh
	WTeam_inputs = W_for.copy() + W_aga.copy()
	LTeam_inputs = L_for.copy() + L_aga.copy()

	
	
	return new_input_row_1, new_input_row_2, WTeam_inputs, LTeam_inputs









def update_dictionary_entry_detailed(team_current_entry, team_stats, 
									 oppt_current_entry, oppt_stats,
									 team_win, games_kept=5
									):
	'''
	entry = [
	0 - num_games,
	
	1 - [[stats for],...], 
	2 - [[stats_against],...], 
	
	3 - [recent_wins?(0,1,1,...)],
	
	4 - [[team_vs_oppt_avg],...], 
	5 - [[oppt_vs_team_avg],...]
	]
	
	
	
	given dictionary
	modify or create entries
	
	
	
	'''
	new_team_entry = deepcopy(team_current_entry)
	new_oppt_entry = deepcopy(oppt_current_entry)
	
	if team_win == 1: oppt_win = 0
	else:             oppt_win = 1
		
		
	
	# how did each team do compared to their normal for/allowed?
	# -----------------------------------------------------------
	# (how do teams perform compared to their own averages versus you)
	'''
	we usually get these stats....
	that team usually gives these stats....
	
	
	
	what did we give up versus what our opponent usually gets
	teams do worse than usual offensively vs us (+ is good)
	= (oppt average stats for) - (oppt game stats)
	so if they usually get 100 points, and they got 80 against us
	= +20
	
	
	what did we get versus what our opponent usually gives up
	= (team game stats) - (oppt average stats aga)
	so if they usually give up 80 points, and we got 100 vs them
	= +20
	
	'''
	team_stats_n = np.array(team_stats.copy())
	oppt_stats_n = np.array(oppt_stats.copy())
	
	# team
	# -----
	oppt_for     = np.sum(np.array(new_oppt_entry[1]), axis=0)
	oppt_allowed = np.sum(np.array(new_oppt_entry[2]), axis=0)
	
	oppt_vs_oppt_avg = np.subtract(    oppt_for, oppt_stats_n)
	team_vs_oppt_avg = np.subtract(team_stats_n, oppt_allowed)
	
	
	# oppt
	# -----
	team_for     = np.sum(np.array(new_team_entry[1]), axis=0)
	team_allowed = np.sum(np.array(new_team_entry[2]), axis=0)
	
	team_vs_team_avg = np.subtract(    team_for, team_stats_n)
	oppt_vs_team_avg = np.subtract(oppt_stats_n, team_allowed)
	

	
	# (only track the X most recent games)
	
	# team
	# ------------------------------
	team_games = new_team_entry[0]
	if team_games >= games_kept:
		# stats FOR
		new_team_entry[1].pop(0)
		new_team_entry[1].append(team_stats)
		
		# stats AGAINST
		new_team_entry[2].pop(0)
		new_team_entry[2].append(oppt_stats)
		
		# win/loss
		new_team_entry[3].pop(0)
		new_team_entry[3].append(team_win)
		
	else:
		new_team_entry[1].append(team_stats      ) # stats FOR
		new_team_entry[2].append(oppt_stats      ) # stats AGAINST
		new_team_entry[3].append(team_win        ) # win/loss
		new_team_entry[0] += 1
		
		
	if len(new_team_entry[4]) >= games_kept:
		# what we got vs what opponent usually gives up
		new_team_entry[4].pop(0)
		new_team_entry[4].append(team_vs_oppt_avg)
		
		# what we gave up vs what opponent usualy gets
		new_team_entry[5].pop(0)
		new_team_entry[5].append(oppt_vs_oppt_avg)
		
	else:
		new_team_entry[4].append(team_vs_oppt_avg) # what we got vs what opponent usually gives up
		new_team_entry[5].append(oppt_vs_oppt_avg) # what we gave up vs what opponent usualy gets
		
		
		
		
	# opponent
	# ------------------------------
	oppt_games = new_oppt_entry[0]
	if oppt_games >= games_kept:
		# stats FOR
		new_oppt_entry[1].pop(0)
		new_oppt_entry[1].append(oppt_stats)
		
		# stats AGAINST
		new_oppt_entry[2].pop(0)
		new_oppt_entry[2].append(team_stats)
		
		# win/loss
		new_oppt_entry[3].pop(0)
		new_oppt_entry[3].append(oppt_win)
	
	else:
		new_oppt_entry[1].append(oppt_stats      ) # stats FOR
		new_oppt_entry[2].append(team_stats      ) # stats AGAINST
		new_oppt_entry[3].append(oppt_win        ) # win/loss
		new_oppt_entry[0] += 1
		
	
	if len(new_oppt_entry[4]) >= games_kept:
		# what we got vs what opponent usually gives up
		new_oppt_entry[4].pop(0)
		new_oppt_entry[4].append(oppt_vs_team_avg)
		
		# what we gave up vs what opponent usualy gets
		new_oppt_entry[5].pop(0)
		new_oppt_entry[5].append(team_vs_team_avg)
		
	else:
		new_oppt_entry[4].append(oppt_vs_team_avg) # what we got vs what opponent usually gives up
		new_oppt_entry[5].append(team_vs_team_avg) # what we gave up vs what opponent usualy gets
		
	
		
	return new_team_entry, new_oppt_entry
	









def update_dictionary_entry(team_dictionary, W_id, L_id, W_stats, L_stats, games_kept=5):
	'''
	entry = [
	0 - num_games,
	
	1 - [[stats for],...], 
	2 - [[stats_against],...], 
	
	3 - [recent_wins?(0,1,1,...)],
	
	4 - [[team_vs_oppt_avg],...], 
	5 - [[oppt_vs_team_avg],...]
	]
	
	'''

	if W_id in team_dictionary and L_id in team_dictionary:
		# call above functions
		team_current_entry = team_dictionary[W_id]
		oppt_current_entry = team_dictionary[L_id]
		
		new_team_entry, new_oppt_entry = update_dictionary_entry_detailed(team_current_entry, W_stats,
																		  oppt_current_entry, L_stats,
																		  1, games_kept)
		team_dictionary[W_id] = new_team_entry
		team_dictionary[L_id] = new_oppt_entry


	elif W_id in team_dictionary:
		team_dictionary[L_id] = [1, [L_stats], [W_stats], [0], [], []]
		
		#team_dictionary[W_id] = [1, [W_stats], [L_stats], [1], [], []]
		team_dictionary[W_id][0] += 1
		team_dictionary[W_id][1].append(W_stats)
		team_dictionary[W_id][2].append(L_stats)
		team_dictionary[W_id][3].append(1)
		
		
	elif L_id in team_dictionary:
		team_dictionary[W_id] = [1, [W_stats], [L_stats], [1], [], []]
		
		#team_dictionary[L_id] = [1, [L_stats], [W_stats], [0], [], []]
		team_dictionary[L_id][0] += 1
		team_dictionary[L_id][1].append(L_stats)
		team_dictionary[L_id][2].append(W_stats)
		team_dictionary[L_id][3].append(0)
		
		
	else:
		team_dictionary[W_id] = [1, [W_stats], [L_stats], [1], [], []]
		team_dictionary[L_id] = [1, [L_stats], [W_stats], [0], [], []]
	
	




def create_team_dict_and_input_rows(df_columns, num_kept=5, mens=True, print_report=False, existing_dict=False):

	# recent games kept in history 
	# -----------------------------
	#num_kept = 5


	# change df to list of rows
	# --------------------------

	# Make a list of rows too
	df_rows = []
	for i in range(len(df_columns[0])):
		new_row = []
		for j in range(len(df_columns)):
			new_row.append(df_columns[j][i])
		df_rows.append(new_row)



	# game solutions and team dictionary
	# -----------------------------------

	# inputs and solutions
	inputs      = []
	solutions   = []


	# team1 ID, team2 ID, team1 win? (1 or 0)
	game_solutions = []


	# key = teamID
	# value = [count, team]
	if existing_dict == False:	team_dictionary = {}
	else:                      	team_dictionary = existing_dict


	for i in range(len(df_rows)):
		row = df_rows[i]
		
		W_id = row[2]
		L_id = row[4]


		if mens:
			WTeam_stats = [row[3]] + row[ 8:22].copy()
			LTeam_stats = [row[5]] + row[22:  ].copy()
		else:
			WTeam_stats = [row[3]] + row[ 8:21].copy()
			LTeam_stats = [row[5]] + row[21:  ].copy()


		# if both teams are in the team dictionary, create input and solution rows
		# -------------------------------------------------------------------------
		if W_id in team_dictionary and L_id in team_dictionary:
			if len(team_dictionary[W_id][4]) > 0 and len(team_dictionary[L_id][4]) > 0:
			
				# create input rows
				# ------------------
				W_data = team_dictionary[W_id]
				L_data = team_dictionary[L_id]

				new_input_row_1, new_input_row_2, WTeam_inputs, LTeam_inputs = create_input_rows(W_data, L_data, womens=(not mens))


				# for solutions
				new_solutions_row_1 = []
				new_solutions_row_2 = []

				# for updating dictionaries
				#W_games = team_dictionary[W_id][0]
				#L_games = team_dictionary[L_id][0]


				# 26 long
				for j in range(len(WTeam_inputs)):
					# team FOR is positive
					if j < (len(WTeam_inputs)//2): # half
						# input and solution rows
						# ------------------------
						new_solutions_row_1.append(WTeam_stats[j])
						new_solutions_row_2.append(LTeam_stats[j])


					# team AGAINST is positive
					else:
						# input and solution rows
						# ------------------------
						new_solutions_row_1.append(-LTeam_stats[j-(len(WTeam_inputs)//2)])
						new_solutions_row_2.append(-WTeam_stats[j-(len(WTeam_inputs)//2)])



				# append new inputs and solutions to overall list
				# ------------------------------------------------
				#inputs.append(new_input_row_1)
				#inputs.append(new_input_row_2)
				inputs.append([W_id, L_id] + new_input_row_1)
				inputs.append([L_id, W_id] + new_input_row_2)

				solutions.append(new_solutions_row_1)
				solutions.append(new_solutions_row_2)

		
		# update team dictionary
		# -----------------------
		update_dictionary_entry(team_dictionary, 
								W_id, L_id, 
								WTeam_stats, LTeam_stats, 
								num_kept)
	

	# print data and return
	# -----------------------
	if print_report:

		print(len(inputs), len(inputs[0]))

		if mens:
			headers_list = ["Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FT%", "FTA", "OR", 
							"DR", "Ast", "TO", "Stl", "Blk", "PF", "Str", "recW%",
							"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFT%", "xFTA", "xOR",
							"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF", "xStr"]
		else:
			headers_list = ["Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FT%", "FTA", "OR", 
							"DR", "Ast", "TO", "Stl", "Blk", "PF", "recW%",
							"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFT%", "xFTA", "xOR",
							"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF"]


		solutions0 = inputs[1][2:]
		#solutions0 = solutions[0]


		lines = ["","","",""]
		for i in range(len(headers_list)):
			if i < len(headers_list)//2:
				lines[0] += "{:>5}  ".format(headers_list[i])
				lines[1] += "{:>5.2}  ".format(solutions0[i])
			else:
				lines[2] += "{:>5}  ".format(headers_list[i])
				lines[3] += "{:>5.2}  ".format(solutions0[i])
				
		for line in lines:
			print(line)


	# return the teams dictionary
	return team_dictionary, inputs, solutions





def prepare_inputs_for_csv(inputs, solutions, mens=True, print_report=False):
	# inputs AND solutions


	# get the headers for the dataframe
	# ----------------------------------
	if mens:
		headers_output = [
			"team1_id", "team2_id",
			
			
			
			"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FT%", "FTA", "OR", 
			"DR", "Ast", "TO", "Stl", "Blk", "PF", "Str",
			
			#"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FTA", "OR", 
			#"DR", "Ast", "TO", "Stl", "Blk", "PF",
			
			"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FTA", "OR", 
			"DR", "Ast", "TO", "Stl", "Blk", "PF",

			"Pts_a", "FGM_a", "FGA_a", "FGM3_a", "FGA3_a", "FTA_a", "OR_a", 
			"DR_a", "Ast_a", "TO_a", "Stl_a", "Blk_a", "PF_a", "Str_a", 
			
			"recW%",
			

			
			
			
			"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFT%", "xFTA", "xOR",
			"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF", "xStr",
			
			#"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFTA", "xOR",
			#"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",
			
			"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFTA", "xOR",
			"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",

			"xPts_a", "xFGM_a", "xFGA_a", "xFGM3_a", "xFGA3_a", "xFTA_a", "xOR_a", 
			"xDR_a", "xAst_a", "xTO_a", "xStl_a", "xBlk_a", "xPF_a", "xStr_a",
			
			"xrecW%",
			



			"solution"]

	else:
		headers_output = [
			"team1_id", "team2_id",
			
			
			
			"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FT%", "FTA", "OR", 
			"DR", "Ast", "TO", "Stl", "Blk", "PF",
			
			#"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FTA", "OR", 
			#"DR", "Ast", "TO", "Stl", "Blk", "PF",
			
			"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FTA", "OR", 
			"DR", "Ast", "TO", "Stl", "Blk", "PF",

			"Pts_a", "FGM_a", "FGA_a", "FGM3_a", "FGA3_a", "FTA_a", "OR_a", 
			"DR_a", "Ast_a", "TO_a", "Stl_a", "Blk_a", "PF_a",
			
			"recW%",
			

			
			
			
			"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFT%", "xFTA", "xOR",
			"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",
			
			#"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFTA", "xOR",
			#"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",
			
			"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFTA", "xOR",
			"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",
			
			"xPts_a", "xFGM_a", "xFGA_a", "xFGM3_a", "xFGA3_a", "xFTA_a", "xOR_a", 
			"xDR_a", "xAst_a", "xTO_a", "xStl_a", "xBlk_a", "xPF_a",

			"xrecW%",
			


			"solution"]





	# solutions as 1 or -1
	# ---------------------
	# to get new solutions, use the points (0,15)
	new_solutions = []
	for row in solutions:
		if abs(row[0]) > abs(row[15]): new_solutions.append([ 1])
		else:                          new_solutions.append([-1])
			

	# convert the inputs to float32 for less storage space
	# -----------------------------------------------------
	new_inputs = []
	for i in range(len(inputs)):
		new_input_row = []
		for j in range(len(inputs[i])):
			new_input_row.append(np.float32(inputs[i][j]))
		new_inputs.append(new_input_row)
			
	#new_inputs = inputs


	# put into the same list
	# -----------------------
	output_rows = []
	for i in range(len(new_inputs)):
		new_row = new_inputs[i]
		#print(len(new_row))
		#print(new_row)
		new_row.append(new_solutions[i][0])

		output_rows.append(new_row)
		

	# print report and return
	# -------------------------
	if print_report:
		print( "headers length: {} \n".format(len(headers_output)))

		print("new_solutions   | rows: {}, columns: {}".format(len(new_solutions),  len(new_solutions[0])))
		print("new_inputs      | rows: {}, columns: {}".format(len(new_inputs),     len(new_inputs[0])))
		print("output_rows     | rows: {}, columns: {}".format(len(output_rows),    len(output_rows[0])))


	return output_rows, headers_output








def submission_input_data(sample_submission_df, team_dictionary, mens=True, print_report=False):

	# get the headers for the dataframe
	# ----------------------------------
	if mens:
		headers = [
			"team1_id", "team2_id",
			
			
			
			"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FT%", "FTA", "OR", 
			"DR", "Ast", "TO", "Stl", "Blk", "PF", "Str",
			
			#"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FTA", "OR", 
			#"DR", "Ast", "TO", "Stl", "Blk", "PF",
			
			"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FTA", "OR", 
			"DR", "Ast", "TO", "Stl", "Blk", "PF",

			"Pts_a", "FGM_a", "FGA_a", "FGM3_a", "FGA3_a", "FTA_a", "OR_a", 
			"DR_a", "Ast_a", "TO_a", "Stl_a", "Blk_a", "PF_a", "Str_a", 
			
			"recW%",
			

			
			
			
			"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFT%", "xFTA", "xOR",
			"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF", "xStr",
			
			#"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFTA", "xOR",
			#"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",
			
			"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFTA", "xOR",
			"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",

			"xPts_a", "xFGM_a", "xFGA_a", "xFGM3_a", "xFGA3_a", "xFTA_a", "xOR_a", 
			"xDR_a", "xAst_a", "xTO_a", "xStl_a", "xBlk_a", "xPF_a", "xStr_a",
			
			"xrecW%"
			

			]

	else:
		headers = [
			"team1_id", "team2_id",
			
			
			
			"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FT%", "FTA", "OR", 
			"DR", "Ast", "TO", "Stl", "Blk", "PF",
			
			#"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FTA", "OR", 
			#"DR", "Ast", "TO", "Stl", "Blk", "PF",
			
			"Pts", "FG%2", "FGA2", "FG%3", "FGA3", "FTA", "OR", 
			"DR", "Ast", "TO", "Stl", "Blk", "PF",

			"Pts_a", "FGM_a", "FGA_a", "FGM3_a", "FGA3_a", "FTA_a", "OR_a", 
			"DR_a", "Ast_a", "TO_a", "Stl_a", "Blk_a", "PF_a",
						
			"recW%",
			

			
			"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFT%", "xFTA", "xOR",
			"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",
			
			#"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFTA", "xOR",
			#"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",
			
			"xPts", "xFG%2", "xFGA2", "xFG%3", "xFGA3", "xFTA", "xOR",
			"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF",

			"xPts_a", "xFGM_a", "xFGA_a", "xFGM3_a", "xFGA3_a", "xFTA_a", "xOR_a", 
			"xDR_a", "xAst_a", "xTO_a", "xStl_a", "xBlk_a", "xPF_a",
			
			"xrecW%"
			

			]



	# get rows from the sample submission
	# ------------------------------------
	matchup_column = list(sample_submission_df['ID'])



	# use matchup column and team dict to create the final inputs
	# ------------------------------------------------------------
	final_inputs = []
	for i in range(len(matchup_column)):
		# get the ids
		# ------------
		values = matchup_column[i].split('_')
		year   = values[0]
		
		WTeam_ID = "{}_{}".format(values[1], year)
		LTeam_ID = "{}_{}".format(values[2], year)
		
		if WTeam_ID in team_dictionary and LTeam_ID in team_dictionary:
			
			# create input rows
			# ------------------
			W_data = team_dictionary[WTeam_ID]
			L_data = team_dictionary[LTeam_ID]
			
			new_input_row_1, new_input_row_2, WTeam_inputs, LTeam_inputs = create_input_rows(W_data, L_data, womens=(not mens))
			
			
			# add in team IDS
			final_inputs.append([WTeam_ID, LTeam_ID] + new_input_row_1)
			final_inputs.append([LTeam_ID, WTeam_ID] + new_input_row_2)
			
		




	# print report and return
	# -------------------------
	if print_report:
		print( "headers length: {} \n".format(len(headers)))

		print("matchup_column  | rows: {}, columns: {}".format(len(matchup_column),  1))
		print("final_inputs    | rows: {}, columns: {}".format(len(final_inputs),     len(final_inputs[0])))


	return final_inputs, headers


















































































































































# end