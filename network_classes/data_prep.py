# data_prep.py



from copy import deepcopy

import random
import time
import threading
import math






#########################################################################################################################
#										PARSE DATA FROM BASIC STATS INTO A DF											#
#########################################################################################################################


# Get each teams stats from a given season
# -----------------------------------------
def stats_for_season(stats_df, season, print_report=False):
    
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
                line0 += "{:>5}, ".format(row[1][j])
                line1 += "{:>5}, ".format(row[2][j])
            
            line0 = line0[:-2] + "] - {} values".format(len(row[1]))
            line1 = line1[:-2] + "] - {} values".format(len(row[1]))
            
            
            print("  number of games played by team: {} \n".format(row[0]))
            print(header_line0, "\n", line0, "\n")
            print(header_line1, "\n", line1)

        
    # game solutions, team dict, and headers_list
    # --------------------------------------------
    return game_solutions, team_dictionary, [headers_list0, headers_list1]







#########################################################################################################################
#													CUSTOM STATS														#
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
#										FINAL PREPARATION FOR DATA AS INPUTS											#
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