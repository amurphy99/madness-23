# data_prep.py



from copy import deepcopy
from statistics import stdev



import random
import time
import threading
import math






#########################################################################################################################
#                                       PARSE DATA FROM BASIC STATS INTO A DF                                           #
#########################################################################################################################


# Get each teams stats from a given season
# -----------------------------------------
def stats_for_season_v0(stats_df, season, print_report=False):
	
	# get just the games from the specified season
	# ---------------------------------------------
	given_df = stats_df[stats_df.Season == season]
	

	# change df to list of columns
	# -----------------------------
	# Creates list of the column names from the df
	columns = given_df.columns.tolist()

	# Creates list of the columns
	df_columns = []
	for expected_column in columns:
		temp = given_df[expected_column].tolist()
		df_columns.append(temp)



	# Normalizing each input stat to 0-1
	# -----------------------------------
	max_columns = []
	for column in df_columns:
		if type(column[0]) != str:
			column_max = max(column)
			max_columns.append(column_max)
			for i in range(len(column)):
				column[i] = (column[i]/column_max)





	# change df to list of rows
	# --------------------------
	# Make a list of rows too
	df_rows = []
	for i in range(len(df_columns[0])):
		new_row = []
		for j in range(len(df_columns)):
			new_row.append(df_columns[j][i])
		df_rows.append(new_row)

	
	# came solutions and team dictionary
	# -----------------------------------
	
	game_solutions = []
	# team1 ID, team2 ID, team1 win? (1 or 0)

	team_dictionary = {}
	# key = teamID
	# value = [count, team]

	for i in range(len(df_rows)):
		row = df_rows[i]

		# game_solutions
		# ---------------
		game_solutions.append( [row[2], row[4], 1] )


		# team dictionary
		# ----------------
		#team_for     = [row[3]] + row[ 8:21].copy()
		#team_allowed = [row[5]] + row[21:  ].copy()
		team_for     = row[ 8:21].copy()
		team_allowed = row[21:  ].copy()


		# WTEAM
		if row[2] not in team_dictionary:
			team_dictionary[row[2]] = [1, team_for, team_allowed]
		else:
			# count
			team_dictionary[row[2]][0] += 1
			# team for
			for j in range(len(team_for)):
				team_dictionary[row[2]][1][j] += team_for[j]
				team_dictionary[row[2]][2][j] += team_allowed[j]

		# LTEAM
		if row[4] not in team_dictionary:
			team_dictionary[row[4]] = [1, team_allowed, team_for]
		else:
			# count
			team_dictionary[row[4]][0] += 1
			# team for
			for j in range(len(team_for)):
				team_dictionary[row[4]][1][j] += team_allowed[j]
				team_dictionary[row[4]][2][j] += team_for[j]
				


	# headers list
	# -------------
	headers_list0 = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", 
					  "DR", "Ast",   "TO",  "Stl", "Blk",  "PF"         ]
	
	headers_list1 = ["xFGM", "xFGA", "xFGM3", "xFGA3", "xFTM", "xFTA", "xOR", 
					  "xDR", "xAst",   "xTO",  "xStl", "xBlk",  "xPF"        ]


	
	# print report
	# -------------
	if print_report:
		print("data size: ")
		print("-----------")
		print("number of games for {:>} season:  {} ".format(season, len(given_df)))
		print("number of columns in base data:   {} ".format(len(df_columns)))
		print("number of rows:                   {} ".format(len(df_rows)))
		
		
		# print column indicis
		# ---------------------
		preview_row = ""
		count       = 0
		for i in range(len(columns)):
			temp         = "{:>3}: {}".format(i, columns[i])
			preview_row += "{:<15}"   .format(temp)
			count       += 1

			if count > 6:
				preview_row += ("\n")
				count        = 0

		print("\n")
		print("column titles and indicis: ")
		print("---------------------------")
		print(preview_row)
		

		# print report on the collected stats
		# ------------------------------------
		print("\n")
		print("base stats report: ")
		print("-------------------")
		print("game_solutions length:  {:>6} games".format(len( game_solutions  )))
		print("team_dictionary length: {:>6} teams".format(len( team_dictionary )))
		print()
		print("team_dictionary[team_id] = [# of games played, [stats FOR totals], [stats AGAINST totals]]")

		
		# teams_dictionary preview
		# -------------------------
		print("\n")
		print("example row:    ")
		print("------------- \n")
		
		
		# headers
		header_line0 = "{:>5}   ".format("id")
		header_line1 = "{:>5}   ".format("id")
		for i in range(len(headers_list0)):
			header_line0 += "{:>5}  ".format(headers_list0[i])
			header_line1 += "{:>5}  ".format(headers_list1[i])

		# data samples
		for i in range(1):
			key = list(team_dictionary.keys())[i]
			row = team_dictionary[ key ]

			line0 = "{:>4}: [".format( key )
			line1 = "{:>4}: [".format( key )
			for j in range(len(row[1])):
				line0 += "{:>5}, ".format( round(row[1][j], 2) )
				line1 += "{:>5}, ".format( round(row[2][j], 2) )
			
			line0 = line0[:-2] + "] - {} values".format(len(row[1]))
			line1 = line1[:-2] + "] - {} values".format(len(row[1]))
			
			
			print("  number of games played by team: {} \n".format(row[0]))
			print(header_line0, "\n", line0, "\n")
			print(header_line1, "\n", line1)

		
	# game solutions, team dict, and headers_list
	# --------------------------------------------
	return game_solutions, team_dictionary, [headers_list0, headers_list1]








# Get each teams stats from a given season
# -----------------------------------------
def inputs_outputs_for_season(stats_df, season, print_report=False):
	
	# get just the games from the specified season
	# ---------------------------------------------
	given_df = stats_df[stats_df.Season == season]

	# change df to list of columns
	# -----------------------------
	columns = given_df.columns.tolist()

	# Creates list of the columns
	df_columns = []
	for expected_column in columns:
		temp = given_df[expected_column].tolist()
		df_columns.append(temp)




	# Normalizing each input stat to 0-1
	# -----------------------------------

	# create columns of all actual data
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
			if value < (column_averages[i] + (column_std[i]*2.5)) and value > (column_averages[i] - (column_std[i]*2.5)):
				within_range_column.append(value)
		within_range.append(within_range_column)


	# now create max columns list for use
	max_columns = []
	for i in range(len(within_range)):
		max_columns.append( max(within_range[i]) )
	max_columns += max_columns


	# normalize all values
	normalized_indicis = [3,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
						  5, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
	for i in range(len(normalized_indicis)):
		column_max  = max_columns[i]
		column      = normalized_indicis[i]
		for i in range(len( df_columns[column] )):
			df_columns[column][i] = (df_columns[column][i]/column_max)


	# create list of normalized variance values
	variance = []
	for i in range(len(column_std)):
		variance.append( (column_std[i]/max_columns[i])**2 )
	variance += variance


	# return max columns and variance lists
	output_max_columns  = max_columns.copy()
	output_variance     = variance.copy()





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
	team_dictionary = {}


	for i in range(len(df_rows)):
		row = df_rows[i]

		WTeam_stats = [row[3]] + row[ 8:21].copy()
		LTeam_stats = [row[5]] + row[21:  ].copy()


		# if both teams are in the team dictionary, create input and solution rows
		# -------------------------------------------------------------------------
		if row[2] in team_dictionary and row[4] in team_dictionary:
			



			new_input_row_1 = team_dictionary[row[2]][1].copy() + team_dictionary[row[4]][2].copy()
			new_input_row_2 = team_dictionary[row[4]][1].copy() + team_dictionary[row[2]][2].copy()

			orig_length = len(new_input_row_1)
			for j in range(len(new_input_row_1)):
				new_input_row_1.append(-new_input_row_2[j])
				new_input_row_2.append(-new_input_row_1[j])


			# for inputs
			#new_input_row_1     = []
			#new_input_row_2     = []

			WTeam_inputs = team_dictionary[row[2]][1].copy() + team_dictionary[row[2]][2].copy()
			LTeam_inputs = team_dictionary[row[4]][1].copy() + team_dictionary[row[4]][2].copy()

			# for solutions
			new_solutions_row_1 = []
			new_solutions_row_2 = []

			# for updating dictionaries
			W_games = team_dictionary[row[2]][0]
			L_games = team_dictionary[row[4]][0]

			# 26 long
			for j in range(len(WTeam_inputs)):
				# team FOR is positive
				if j < (len(WTeam_inputs)//2): # half

					# input and solution rows
					# ------------------------
					#new_input_row_1.append(  WTeam_inputs[j]-LTeam_inputs[j] )
					#new_input_row_2.append( -WTeam_inputs[j]+LTeam_inputs[j] )

					new_solutions_row_1.append(WTeam_stats[j])
					new_solutions_row_2.append(LTeam_stats[j])

					# Update team dictionaries
					# -------------------------
					# WTeam
					team_dictionary[row[2]][1][j] = ((W_games * team_dictionary[row[2]][1][j]) + WTeam_stats[j])/(W_games+1)
					team_dictionary[row[2]][2][j] = ((W_games * team_dictionary[row[2]][2][j]) + LTeam_stats[j])/(W_games+1)

					# LTeam
					team_dictionary[row[4]][1][j] = ((L_games * team_dictionary[row[4]][1][j]) + LTeam_stats[j])/(L_games+1)
					team_dictionary[row[4]][2][j] = ((L_games * team_dictionary[row[4]][2][j]) + WTeam_stats[j])/(L_games+1)

				# team AGAINST is positive
				else:
					# input and solution rows
					# ------------------------
					#new_input_row_1.append( -WTeam_inputs[j]+LTeam_inputs[j] )
					#new_input_row_2.append(  WTeam_inputs[j]-LTeam_inputs[j] )

					new_solutions_row_1.append(-LTeam_stats[j-(len(WTeam_inputs)//2)])
					new_solutions_row_2.append(-WTeam_stats[j-(len(WTeam_inputs)//2)])


			# update game count of both teams
			team_dictionary[row[2]][0] += 1
			team_dictionary[row[4]][0] += 1


			# append new inputs and solutions to overall list
			# ------------------------------------------------
			inputs.append(new_input_row_1)
			inputs.append(new_input_row_2)

			solutions.append(new_solutions_row_1)
			solutions.append(new_solutions_row_2)





		# if both teams are not in the dictionary already then just add/update them
		# --------------------------------------------------------------------------
		# create LTeam entry, update WTeam entry
		elif row[2] in team_dictionary:
			# LTeam
			team_dictionary[row[4]] = [1, WTeam_stats, LTeam_stats]

			# WTeam
			W_games = team_dictionary[row[2]][0]
			for j in range(len(WTeam_stats)):
				team_dictionary[row[2]][1][j] = ((W_games * team_dictionary[row[2]][1][j]) + WTeam_stats[j])/(W_games+1)
				team_dictionary[row[2]][2][j] = ((W_games * team_dictionary[row[2]][2][j]) + LTeam_stats[j])/(W_games+1)
			team_dictionary[row[2]][0] += 1


		# create WTeam entry, update LTeam entry
		elif row[4] in team_dictionary:
			# WTeam
			team_dictionary[row[2]] = [1, LTeam_stats, WTeam_stats]
			
			# LTeam
			L_games = team_dictionary[row[4]][0]
			for j in range(len(LTeam_stats)):
				team_dictionary[row[4]][1][j] = ((L_games * team_dictionary[row[4]][1][j]) + LTeam_stats[j])/(L_games+1)
				team_dictionary[row[4]][2][j] = ((L_games * team_dictionary[row[4]][2][j]) + WTeam_stats[j])/(L_games+1)
			team_dictionary[row[4]][0] += 1


		# create entry for both WTeam and LTeam
		else:
			team_dictionary[row[2]] = [1, WTeam_stats, LTeam_stats]
			team_dictionary[row[4]] = [1, LTeam_stats, WTeam_stats]




	# after this loop i have:
	# * inputs and solutions for the given year




	# headers list
	# -------------
	headers_list0 = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", 
					  "DR", "Ast",   "TO",  "Stl", "Blk",  "PF"         ]
	
	headers_list1 = ["xFGM", "xFGA", "xFGM3", "xFGA3", "xFTM", "xFTA", "xOR", 
					  "xDR", "xAst",   "xTO",  "xStl", "xBlk",  "xPF"        ]


	# print report
	# -------------
	headers_list = ["Pts", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", 
					"DR", "Ast", "TO", "Stl", "Blk", "PF", 
					"xPts", "xFGM", "xFGA", "xFGM3", "xFGA3", "xFTM", "xFTA", "xOR",
					"xDR", "xAst", "xTO", "xStl", "xBlk", "xPF"]

	print(len(inputs), len(solutions))

	line1 = ""
	line2 = ""
	for i in range(len(output_variance)//2):
		line1 += "{:>6} ".format(headers_list[i])
		line2 += "{:>6}%".format( round(output_variance[i]*100, 2) )
	print(line1)
	print(line2)


	return inputs, solutions, output_max_columns, output_variance













def separate_fga_and_fgp(stats_df, season, print_report=False):
	# get just the games from the specified season
	# ---------------------------------------------
	given_df = stats_df[stats_df.Season == season]

	# change df to list of columns
	# -----------------------------
	columns = given_df.columns.tolist()

	# Creates list of the columns
	orig_columns = []
	for expected_column in columns:
		temp = given_df[expected_column].tolist()
		orig_columns.append(temp)

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
		number_columns[ 1][i] = number_columns[ 1][i]/number_columns[ 2][i] #  1: WFGM2 /  2: WFGA2
		number_columns[ 3][i] = number_columns[ 3][i]/number_columns[ 4][i] #  3: WFGM3 /  4: WFGA3
		if number_columns[ 6][i] == 0: 
			number_columns[ 5][i] = 1.0
		else:
			number_columns[ 5][i] = number_columns[ 5][i]/number_columns[ 6][i] #  5: WFTM  /  6: WFTA

		number_columns[15][i] = number_columns[15][i]/number_columns[16][i] # 15: LFGM2 / 16: LFGA2
		number_columns[17][i] = number_columns[17][i]/number_columns[18][i] # 17: LFGM3 / 18: LFGA3
		if number_columns[20][i] == 0:
			number_columns[19][i] = 1.0
		else:
			number_columns[19][i] = number_columns[19][i]/number_columns[20][i] # 19: LFTM  / 20: LFTA

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


	return inputs_outputs_for_season(number_columns, team_IDs, headers_list, season, print_report)






# Get each teams stats from a given season
# -----------------------------------------
def inputs_outputs_for_season(input_columns, team_IDs, headers_list, season, print_report=False):
	
	# get just the games from the specified season
	# change df to list of columns
	# ---------------------------------------------
	df_columns = input_columns


	# Normalizing each input stat to 0-1
	# -----------------------------------

	# create columns of all actual data
	split = int(len(df_columns)//2)
	stats_columns = deepcopy(df_columns[     :split])
	additional    = deepcopy(df_columns[split:     ])

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
			if value < (column_averages[i] + (column_std[i]*2.5)) and value > (column_averages[i] - (column_std[i]*2.5)):
				within_range_column.append(value)
		within_range.append(within_range_column)


	# now create max columns list for use
	max_columns = []
	for i in range(len(within_range)):
		max_columns.append( max(within_range[i]) )
	max_columns += max_columns


	# normalize all values
	for i in range(len(df_columns)):
		column_max  = max_columns[i]
		#print(i, column_max)
		for j in range(len( df_columns[i] )):
			df_columns[i][j] = (df_columns[i][j]/column_max)


	# create list of normalized variance values
	variance = []
	for i in range(len(column_std)):
		variance.append( (column_std[i]/max_columns[i])**2 )
	variance += variance


	# return max columns and variance lists
	output_max_columns  = max_columns.copy()
	output_variance     = variance.copy()





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
	team_dictionary = {}


	for i in range(len(df_rows)):
		row    = df_rows[i]
		id_row = [team_IDs[0][i], team_IDs[1][i]]

		#WTeam_stats = [row[3]] + row[ 8:21].copy()
		#LTeam_stats = [row[5]] + row[21:  ].copy()

		split = int(len(row)//2)
		WTeam_stats = row[     :split].copy()
		LTeam_stats = row[split:  ].copy()



		# if both teams are in the team dictionary, create input and solution rows
		# -------------------------------------------------------------------------
		if id_row[0] in team_dictionary and id_row[1] in team_dictionary:


			new_input_row_1 = team_dictionary[id_row[0]][1].copy() + team_dictionary[id_row[1]][2].copy()
			new_input_row_2 = team_dictionary[id_row[1]][1].copy() + team_dictionary[id_row[0]][2].copy()

			orig_length = len(new_input_row_1)
			for j in range(len(new_input_row_1)):
				new_input_row_1.append(-new_input_row_2[j])
				new_input_row_2.append(-new_input_row_1[j])


			# for inputs
			#new_input_row_1     = []
			#new_input_row_2     = []

			WTeam_inputs = team_dictionary[id_row[0]][1].copy() + team_dictionary[id_row[0]][2].copy()
			LTeam_inputs = team_dictionary[id_row[1]][1].copy() + team_dictionary[id_row[1]][2].copy()

			# for solutions
			new_solutions_row_1 = []
			new_solutions_row_2 = []

			# for updating dictionaries
			W_games = team_dictionary[id_row[0]][0]
			L_games = team_dictionary[id_row[1]][0]

			# 26 long
			for j in range(len(WTeam_inputs)):
				# team FOR is positive
				if j < (len(WTeam_inputs)//2): # half

					# input and solution rows
					# ------------------------
					#new_input_row_1.append(  WTeam_inputs[j]-LTeam_inputs[j] )
					#new_input_row_2.append( -WTeam_inputs[j]+LTeam_inputs[j] )

					new_solutions_row_1.append(WTeam_stats[j])
					new_solutions_row_2.append(LTeam_stats[j])

					# Update team dictionaries
					# -------------------------
					# WTeam
					team_dictionary[id_row[0]][1][j] = ((W_games * team_dictionary[id_row[0]][1][j]) + WTeam_stats[j])/(W_games+1)
					team_dictionary[id_row[0]][2][j] = ((W_games * team_dictionary[id_row[0]][2][j]) + LTeam_stats[j])/(W_games+1)

					# LTeam
					team_dictionary[id_row[1]][1][j] = ((L_games * team_dictionary[id_row[1]][1][j]) + LTeam_stats[j])/(L_games+1)
					team_dictionary[id_row[1]][2][j] = ((L_games * team_dictionary[id_row[1]][2][j]) + WTeam_stats[j])/(L_games+1)

				# team AGAINST is positive
				else:
					# input and solution rows
					# ------------------------
					#new_input_row_1.append( -WTeam_inputs[j]+LTeam_inputs[j] )
					#new_input_row_2.append(  WTeam_inputs[j]-LTeam_inputs[j] )

					new_solutions_row_1.append(-LTeam_stats[j-(len(WTeam_inputs)//2)])
					new_solutions_row_2.append(-WTeam_stats[j-(len(WTeam_inputs)//2)])


			# update game count of both teams
			team_dictionary[id_row[0]][0] += 1
			team_dictionary[id_row[1]][0] += 1


			# append new inputs and solutions to overall list
			# ------------------------------------------------
			inputs.append(new_input_row_1)
			inputs.append(new_input_row_2)


			# add scores onto the end of the solutions
			#for_1 =  2*new_solutions_row_1[ 0]*new_solutions_row_1[ 1] +  3*new_solutions_row_1[ 2]*new_solutions_row_1[ 3] +    new_solutions_row_1[ 4]*new_solutions_row_1[ 5]
			#aga_1 = -2*new_solutions_row_1[13]*new_solutions_row_1[14] + -3*new_solutions_row_1[15]*new_solutions_row_1[16] + -1*new_solutions_row_1[17]*new_solutions_row_1[18]
			#new_solutions_row_1.append(for_1)
			#new_solutions_row_1.append(aga_1)

			#for_2 =  2*new_solutions_row_2[ 0]*new_solutions_row_2[ 1] +  3*new_solutions_row_2[ 2]*new_solutions_row_2[ 3] +    new_solutions_row_2[ 4]*new_solutions_row_2[ 5]
			#aga_2 = -2*new_solutions_row_2[13]*new_solutions_row_2[14] + -3*new_solutions_row_2[15]*new_solutions_row_2[16] + -1*new_solutions_row_2[17]*new_solutions_row_2[18]
			#new_solutions_row_2.append(for_2)
			#new_solutions_row_2.append(aga_2)

			solutions.append(new_solutions_row_1)
			solutions.append(new_solutions_row_2)



		# if both teams are not in the dictionary already then just add/update them
		# --------------------------------------------------------------------------
		# create LTeam entry, update WTeam entry
		elif id_row[0] in team_dictionary:
			# LTeam
			team_dictionary[id_row[1]] = [1, WTeam_stats, LTeam_stats]

			# WTeam
			W_games = team_dictionary[id_row[0]][0]
			for j in range(len(WTeam_stats)):
				team_dictionary[id_row[0]][1][j] = ((W_games * team_dictionary[id_row[0]][1][j]) + WTeam_stats[j])/(W_games+1)
				team_dictionary[id_row[0]][2][j] = ((W_games * team_dictionary[id_row[0]][2][j]) + LTeam_stats[j])/(W_games+1)
			team_dictionary[id_row[0]][0] += 1


		# create WTeam entry, update LTeam entry
		elif id_row[1] in team_dictionary:
			# WTeam
			team_dictionary[id_row[0]] = [1, LTeam_stats, WTeam_stats]
			
			# LTeam
			L_games = team_dictionary[id_row[1]][0]
			for j in range(len(LTeam_stats)):
				team_dictionary[id_row[1]][1][j] = ((L_games * team_dictionary[id_row[1]][1][j]) + LTeam_stats[j])/(L_games+1)
				team_dictionary[id_row[1]][2][j] = ((L_games * team_dictionary[id_row[1]][2][j]) + WTeam_stats[j])/(L_games+1)
			team_dictionary[id_row[1]][0] += 1


		# create entry for both WTeam and LTeam
		else:
			team_dictionary[id_row[0]] = [1, WTeam_stats, LTeam_stats]
			team_dictionary[id_row[1]] = [1, LTeam_stats, WTeam_stats]




	# after this loop i have:
	# * inputs and solutions for the given year




	# headers list
	# -------------
	headers_list0 = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", 
					  "DR", "Ast",   "TO",  "Stl", "Blk",  "PF"         ]
	
	headers_list1 = ["xFGM", "xFGA", "xFGM3", "xFGA3", "xFTM", "xFTA", "xOR", 
					  "xDR", "xAst",   "xTO",  "xStl", "xBlk",  "xPF"        ]


	# print report
	# -------------

	print(len(inputs), len(solutions))

	line1 = ""
	line2 = ""
	line3 = ""
	for i in range(len(output_variance)//2):
		line1 += "{:>6} ".format(headers_list[i])
		line2 += "{:>6} ".format( round(inputs[0][i], 2) )
		line3 += "{:>6}%".format( round(output_variance[i]*100, 2) )
	print(line1)
	print(line2)
	print(line3)


	return inputs, solutions, output_max_columns, output_variance


#########################################################################################################################
#                                                   CUSTOM STATS                                                        #
#########################################################################################################################


# Base Stats + Total Rebound Differential
# ----------------------------------------
def base_w_rebound_diff(team_dictionary, print_report=False):
	'''
	TOTAL REBOUND DIFFERENTIAL
	---------------------------
	* turn (defensive rebounds) and (offensive rebounds) into (average total rebound differential)
	
	'''
	# output dict
	rebound_diff_team_dictionary = deepcopy(team_dictionary)

	# loop
	for i in range(len( list(rebound_diff_team_dictionary.keys()) )):
		#       [      0,           1,               2]
		# row = [team_id, [stats_for], [stats_allowed]]
		row = rebound_diff_team_dictionary[ list(rebound_diff_team_dictionary.keys())[i] ]
		
		# dont do this twice
		if len(row[1]) > 12: 

			# old stats:
			# 0: FGM, 1: FGA, 2: FGM3, 3: FGA3, 4: FTM, 5: FTA, 6: OR, 7: DR, 8: Ast, 9: TO, 10: Stl 11: Blk, 12: PF
			total_rebounds_for     = (row[1][6] + row[1][7])
			total_rebounds_against = (row[2][6] + row[2][7])

			rebound_differential_for     = (total_rebounds_for - total_rebounds_against)
			rebound_differential_against = (total_rebounds_against - total_rebounds_for)

			# pop defensive rebounds
			row[1].pop(7)
			row[2].pop(7)

			# replace offensive rebounds with rebound differential
			row[1][6] = rebound_differential_for
			row[2][6] = rebound_differential_against

			# new_stats:
			# 0: FGM, 1: FGA, 2: FGM3, 3: FGA3, 4: FTM, 5: FTA, 6: RDif, 7: Ast, 8: TO, 9: Stl 10: Blk, 11: PF

	
	# headers list
	# -------------
	headers_list = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "RDif", "Ast", "TO", "Stl", "Blk", "PF"]

	
	# print report
	# -------------
	if print_report:
		print("base_w_rebound_diff() report: ")
		print("------------------------------")

		
		# print report on the collected stats
		# ------------------------------------
		print("team_dictionary length:              {:>6} teams".format(len(team_dictionary             )))
		print("rebound_diff_team_dictionary length: {:>6} teams".format(len(rebound_diff_team_dictionary)))
		print()


		# print example data
		# -------------------
		#  id    FGM   FGA  FGM3  FGA3   FTM   FTA  RDif   Ast    TO   Stl   Blk    PF
		#  23: [1110, 2585,  355, 1112,  573,  797,   76,  568,  560,  282,  164,  746] - 12 values

		# header
		# -------
		header_line = "{:>5}   ".format("id")
		for header in headers_list:
			header_line += "{:>5}  ".format(header)
		print(header_line)

		# data
		# -----
		for i in range(1):
			key = list(rebound_diff_team_dictionary.keys())[i]
			row = rebound_diff_team_dictionary[ key ]

			line = "{:>5}: [".format( key )

			for stat in row[1]:
				line += "{:>5}, ".format(stat)

			line = line[:-2] + "] - {} values".format(len(row[1]))
			print(line, "\n")

		print()
	
		
	# return dict and column headers
	# -------------------------------
	return rebound_diff_team_dictionary, headers_list








# Base Stats + FOR and AGAINST
# -----------------------------
def for_and_against(team_dictionary, print_report=False):
	'''
	STATS FOR AND AGAINST
	---------------------------
	* 
	
	'''
	# output dict
	output_dict = deepcopy(team_dictionary)

	# loop
	for i in range(len( list(output_dict.keys()) )):
		#       [      0,           1,               2]
		# row = [team_id, [stats_for], [stats_allowed]]
		row = output_dict[ list(output_dict.keys())[i] ]
		
		new_row_for     = row[1].copy()
		new_row_against = row[2].copy()
		
		for j in range(len(row[1])):
			new_row_for.append(     -row[2][j] )
			new_row_against.append( -row[1][j] )
			
		row[1] = new_row_for
		row[2] = new_row_against
		
	
	# headers list
	# -------------
	headers_list = [ "FGM",  "FGA",  "FGM3",  "FGA3",  "FTM",  "FTA",  "OR",  "DR",  "Ast",  "TO",  "Stl",  "Blk",  "PF",
					"xFGM", "xFGA", "xFGM3", "xFGA3", "xFTM", "xFTA", "xOR", "xDR", "xAst", "xTO", "xStl", "xBlk", "xPF"]

	
	# print report
	# -------------
	if print_report:
		print("base_w_rebound_diff() report: ")
		print("------------------------------")

		
		# print report on the collected stats
		# ------------------------------------
		print("team_dictionary length:              {:>6} teams".format(len(team_dictionary             )))
		print("rebound_diff_team_dictionary length: {:>6} teams".format(len(output_dict)))
		print()


		# print example data
		# -------------------
		#  id    FGM   FGA  FGM3  FGA3   FTM   FTA  RDif   Ast    TO   Stl   Blk    PF
		#  23: [1110, 2585,  355, 1112,  573,  797,   76,  568,  560,  282,  164,  746] - 12 values

		# header
		# -------
		header_line = "{:>5}   ".format("id")
		for header in headers_list:
			header_line += "{:>5}  ".format(header)
		print(header_line)

		# data
		# -----
		for i in range(1):
			key = list(output_dict.keys())[i]
			row = output_dict[ key ]

			line = "{:>5}: [".format( key )

			for stat in row[1]:
				line += "{:>5}, ".format(stat)

			line = line[:-2] + "] - {} values".format(len(row[1]))
			print(line, "\n")

		print()
	
	
	# return dict and column headers
	# -------------------------------
	return output_dict, headers_list












#########################################################################################################################
#                                       FINAL PREPARATION FOR DATA AS INPUTS                                            #
#########################################################################################################################


# team_dictionary from TOTALS to AVERAGES
# ----------------------------------------
def team_totals_to_averages(team_dictionary, headers_list, print_report=False):
	
	team_averages = {}
	team_keys = list(team_dictionary.keys())

	for key in team_keys:
		entry = team_dictionary[key]
		count = entry[0]

		avg_team_for     = []
		avg_team_allowed = []

		for i in range(len(entry[1])):
			avg_team_for.append( (entry[1][i] / count) )
			avg_team_allowed.append( (entry[2][i] / count) )

		team_averages[key] = [avg_team_for, avg_team_allowed]

		
		
	# print report
	# -------------
	if print_report:
		print("team_totals_to_averages() report: ")
		print("----------------------------------")
		
		
		# print report on the collected stats
		# ------------------------------------
		print("input team_dictionary length: {:>6} teams".format(len(team_dictionary)))
		print("output team_averages length:  {:>6} teams".format(len(team_averages)))
		print()


		# print example data
		# -------------------
		#  id    FGM   FGA  FGM3  FGA3   FTM   FTA  RDif   Ast    TO   Stl   Blk    PF
		#  23: [1110, 2585,  355, 1112,  573,  797,   76,  568,  560,  282,  164,  746] - 12 values

		
		# header
		# -------
		header_line = "{:>5}   ".format("id")
		for header in headers_list:
			header_line += "{:>5}  ".format(header)
		print(header_line)

		
		# data
		# -----
		for i in range(1):
			key = list(team_averages.keys())[i]
			row = team_averages[ key ]

			line = "{:>5}: [".format( key )
			
			for stat in row[0]:
				line += "{:>5}, ".format( round(stat, 1) )

			line = line[:-2] + "] - {} values".format(len(row[0]))
			print(line, "\n")
			
		print()
	
	
	# return new averages dict
	# -------------------------
	return team_averages





# preparing data officially as inputs and solutions
# --------------------------------------------------
# (subtracts all TEAM 2 stats from TEAM 1 stats)
def prep_inputs_and_solutions(data_dict, game_solutions, print_report=False):
	inputs    = []
	solutions = []

	for i in range(len(game_solutions)):
		game  = game_solutions[i]
		team1 = game[0]
		team2 = game[1]

		if random.uniform(0,1) > 0.5:
			#input_row = data_dict[team1][1] + data_dict[team2][1]
			input_row = []
			for j in range(len(data_dict[team1][0])):
				input_row.append( (data_dict[team1][0][j] - data_dict[team2][0][j]) )

			inputs.append(input_row)
			solutions.append(1)

			# second way around
			input_row = []
			for j in range(len(data_dict[team1][0])):
				input_row.append( (data_dict[team2][0][j] - data_dict[team1][0][j]) )

			inputs.append(input_row)
			solutions.append(0)



		else:
			#input_row = data_dict[team2][1] + data_dict[team1][1]
			input_row = []
			for j in range(len(data_dict[team1][0])):
				input_row.append( (data_dict[team2][0][j] - data_dict[team1][0][j]) )

			inputs.append(input_row)
			solutions.append(0)


			# second way around
			input_row = []
			for j in range(len(data_dict[team1][0])):
				input_row.append( (data_dict[team1][0][j] - data_dict[team2][0][j]) )

			inputs.append(input_row)
			solutions.append(1)


	# print report
	# -------------
	if print_report:
		print("prep_inputs_and_solutions() report: ")
		print("------------------------------------")
		
		print("inputs length:    {}".format(len(inputs   )    ))
		print("solutions length: {}".format(len(solutions)    ))
		print()
		print("unique games:     {}".format(len(solutions)//2 ))
		print("(games are tested two times, one with each team as team 1 and team 2)")
		
		print()
		
	
	# return inputs and solutions lists
	# ----------------------------------
	return inputs, solutions

























# end