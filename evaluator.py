from unidecode import unidecode
from itertools import combinations
import numpy as np
import pandas as pd
import os

class Evaluator:
    def stat_greater_than(self, stat, player_data, threshold):
        #returns if player's given statistic is higher than passed in threshold
        return float(player_data[stat].values[0]) > float(threshold)

    def stat_less_than(self, stat, player_data, threshold):
        #returns if player's given statistic is lower than passed in threshold
        return float(player_data[stat].values[0]) < float(threshold)

    def evaluator(self, symbol, first_stat, second_stat, third_stat, player_data, threshold):
        #returns if player achieved above prop threshold if an over bet, under prop threshold if an under bet

        #preprocessing for triple double props
        if third_stat:
            triple = float(player_data[first_stat].values[0]) + float(player_data[second_stat].values[0]) + float(player_data[third_stat].values[0])
            if symbol == '>':
                if triple > threshold:
                    return 'Y'
                return 'N' 
            else:
                if triple < threshold:
                    return 'Y'
                return 'N' 
        
        #preprocessing for double double props
        if second_stat:
            double = float(player_data[first_stat].values[0]) + float(player_data[second_stat].values[0])
            if symbol == '>':
                if double > threshold:
                    return 'Y'
                return 'N'
            else:
                if double < threshold:
                    return 'Y'
                return 'N'

        if symbol == '>':
            if self.stat_greater_than(first_stat, player_data, threshold):
                return 'Y'
            return 'N'
        else:
            if self.stat_less_than(first_stat, player_data, threshold):
                return 'Y'
            return 'N'

    def past_games_stats_evaluator(self, vals, prop_bet_number, is_over):
        #returns array of how many games player performs above or below given threshold
        ovr_avg_correct = []
        if is_over:
            for i in vals:
                if prop_bet_number < i:
                    ovr_avg_correct.append('Y')
                else:
                    ovr_avg_correct.append('N')
            return ovr_avg_correct
        else:
            for i in vals:
                if prop_bet_number > i:
                    ovr_avg_correct.append('Y')
                else:
                    ovr_avg_correct.append('N')
            return ovr_avg_correct
    
    def past_games_trends(self, predictions, all_box_score_results, is_evaluation):
        #returns evaluations if a player performs above or below given threshold for all games played, past 10 games played, past 5 games played
        all_games_prcts, all_games_trues, last5_prcts, last5_trues, last10_prcts, last10_trues = [], [], [], [], [], []
        for i in predictions.index:
            bet = predictions.loc[i]
            bet['Play'] = bet['Play'].replace('  ', ' ')
            name = bet['Play'].split(' ')[0]
            teams = [bet['Teams']]
            prop = bet['Play'].split(' ')[1]

            #determine type of prop bets
            prop_bet_type = bet['Play'].split(' ')[2]
            if 'Yes' in prop_bet_type:
                prop_bet_type = bet['Play'].split(' ')[1]
            if 'Double-Double' in prop_bet_type or 'Triple-Double' in prop_bet_type:
                over = 'Yes' in prop
                prop_bet_number = None
            else:
                over = prop[0] == 'o'
                prop_bet_number = float(prop[1:].split(' ')[0])
            
            #evaluate past trends based on identified player's past statistics 
            matching_name = all_box_score_results[all_box_score_results['name'] == name]
            all_games = matching_name[matching_name['team'].isin(teams)]
            last5_games = all_games[:5]
            last10_games =  all_games[:10]
            all_games_prct, all_games_true = self.past_evaluator(all_games, prop_bet_type, prop_bet_number, over)
            last5_prct, last5_true = self.past_evaluator(last5_games, prop_bet_type, prop_bet_number, over)
            last10_prct, last10_true = self.past_evaluator(last10_games, prop_bet_type, prop_bet_number, over)
            all_games_prcts.append(all_games_prct)
            all_games_trues.append(all_games_true)
            last5_prcts.append(last5_prct)
            last5_trues.append(last5_true)
            last10_prcts.append(last10_prct)
            last10_trues.append(last10_true)  

        predictions['All Game Percentages'], predictions['All Game Correct'], predictions['Last 5 Percentages'], predictions['Last 5 Correct'], predictions['Last 10 Percentages'], predictions['Last 10 Correct']  = all_games_prcts, all_games_trues, last5_prcts, last5_trues, last10_prcts, last10_trues

        #determine if inputs for evaluted predictions which have 'Correct' feature identified or current predictions which are to be determined
        if is_evaluation:
            features_list = ['Play', 'Expert', 'Odds',
            'Units', 'Payout', 'Net Units Record', 'Teams', 'name', 'opponent',
            'hmcrt_adv', 'Profit', 'Correct', 'All Game Percentages',
            'All Game Correct', 'Last 5 Percentages', 'Last 5 Correct',
            'Last 10 Percentages', 'Last 10 Correct']
        else:
            features_list = ['Play', 'Expert', 'Teams',  'opponent',
        'hmcrt_adv', 'All Game Percentages',
        'All Game Correct', 'Last 5 Percentages', 'Last 5 Correct',
        'Last 10 Percentages', 'Last 10 Correct']
        return predictions[features_list]


    def past_games_double_digit_stats_evaluator(self, games, double_type):
        #returns past game evaluations for double-double or triple-double prop types
        all_vals = []
        if double_type == 'Double':
            for comb in [com for com in combinations(['pts', 'ast', 'trb', 'stl', 'blk'], 2)]:
                all_vals.append([v1 > 9 and v2 > 9 for v1, v2 in zip(games[comb[0]].values, games[comb[1]].values)])
        else:
            for comb in [com for com in combinations(['pts', 'ast', 'trb', 'stl', 'blk'], 3)]:
                all_vals.append([v1 > 9 and v2 > 9 and v3 > 9 for v1, v2, v3 in zip(games[comb[0]].values, games[comb[1]].values, games[comb[2]].values)])
        vals = np.any(all_vals, axis = 0)
        return ['Y' if i else 'N' for i in vals]

    def past_evaluator(self, games, prop_bet_type, prop_bet_number, is_over):
        #returns past game evaluations for prop types
        if prop_bet_type == 'Pts':
            vals = games['pts'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Ast':
            vals = games['ast'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Rebs':
            vals = games['trb'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == '3pt':
            vals = games['fg3'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Stl':
            vals = games['stl'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Blk':
            vals = games['blk'].values
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Pts+Ast':
            vals = [v1 + v2 for v1, v2 in zip(games['pts'].values, games['ast'].values)]
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Pts+Rebs':
            vals = [v1 + v2 for v1, v2 in zip(games['pts'].values, games['trb'].values)]
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Rebs+Ast':
            vals = [v1 + v2 for v1, v2 in zip(games['trb'].values, games['ast'].values)]
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Stl+Blk':
            vals = [v1 + v2 for v1, v2 in zip(games['trb'].values, games['ast'].values)]
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)
        if prop_bet_type == 'Pts+Rebs+Ast':
            vals = [v1 + v2 + v3 for v1, v2, v3 in zip(games['pts'].values, games['trb'].values, games['ast'].values)]
            ovr_avg_correct = self.past_games_stats_evaluator(vals, prop_bet_number, is_over)    
        if prop_bet_type == 'Double-Double' or prop_bet_type == 'Triple-Double':
            ovr_avg_correct = self.past_games_double_digit_stats_evaluator(games, prop_bet_type.split('-')[0])
        ovr_avg_correct_prct = sum([1 for i in ovr_avg_correct if i == 'Y']) / len(ovr_avg_correct)
        if ovr_avg_correct_prct > 0.5:
            over_half = 1
        else:
            over_half = 0
        return ovr_avg_correct_prct, over_half 

    def predictions_evaluator(self, predictions, box_score):
        #returns evaluations of past predictions based on past box score corresponding
        correct, names = [], []
        for p in range(len(predictions)):
            bet = predictions.loc[p]
            bet['Play'] = bet['Play'].replace('  ', ' ')
            i = bet['Play']
            name = unidecode(i.split(' ')[0])
            teams = [bet['Teams']]
            box_score['name'] = box_score['name'].apply(unidecode)
            matching_name = box_score.loc[box_score['name'] == name]
            player_data = matching_name[matching_name['team'].isin(teams)]

            #return 'X' if player did not play in past box score
            if len(player_data) == 0:
                correct.append('X')
                continue
            prediction = [j for j in i.split(' ')[1:] if len(j) > 0]
            
            #determine what prop type is and evaluate based on given statistic
            if 'Double-Double' in prediction[0]:
                double_double_check = self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('ast', player_data, 9) or self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('trb', player_data, 9) or self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('trb', player_data, 9) or self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('stl', player_data, 9) or self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('blk', player_data, 9) or self.stat_greater_than('trb', player_data, 9) and self.stat_greater_than('stl', player_data, 9) or self.stat_greater_than('trb', player_data, 9) and self.stat_greater_than('blk', player_data, 9) or self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('stl', player_data, 9) or self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('blk', player_data, 9)
                if 'Yes' in prediction[1]:
                    if double_double_check:
                        correct.append('Y')
                    else:
                        correct.append('N')
            elif 'Triple-Double' in prediction[0]:
                triple_double_check = (self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('trb', player_data, 9)) or (self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('stl', player_data, 9)) or (self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('blk', player_data, 9)) or (self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('trb', player_data, 9) and self.stat_greater_than('stl', player_data, 9)) or (self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('trb', player_data, 9) and self.stat_greater_than('blk', player_data, 9)) or (self.stat_greater_than('pts', player_data, 9) and self.stat_greater_than('stl', player_data, 9) and self.stat_greater_than('blk', player_data, 9)) or (self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('trb', player_data, 9) and self.stat_greater_than('stl', player_data, 9)) or (self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('trb', player_data, 9) and self.stat_greater_than('blk', player_data, 9)) or (self.stat_greater_than('ast', player_data, 9) and self.stat_greater_than('stl', player_data, 9) and self.stat_greater_than('blk', player_data, 9)) or (self.stat_greater_than('trb', player_data, 9) and self.stat_greater_than('stl', player_data, 9) and self.stat_greater_than('blk', player_data, 9))
                if 'Yes' in prediction[1]:
                    if triple_double_check:
                        correct.append('Y')
                    else:
                        correct.append('N')
            elif prediction[0][0] == 'o':
                if prediction[1] == 'Pts':
                    correct.append(self.evaluator('>', 'pts', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Ast':
                    correct.append(self.evaluator('>', 'ast', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Rebs':
                    correct.append(self.evaluator('>', 'trb', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == '3pt':
                    correct.append(self.evaluator('>', 'fg3', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Pts+Ast':
                    correct.append(self.evaluator('>', 'pts', 'ast', None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Pts+Rebs':
                    correct.append(self.evaluator('>', 'pts', 'trb', None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Rebs+Ast':
                    correct.append(self.evaluator('>', 'trb', 'ast', None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Pts+Rebs+Ast':
                    correct.append(self.evaluator('>', 'pts', 'ast', 'trb', player_data, float(prediction[0][1:])))
                if prediction[1] == 'Stl':
                    correct.append(self.evaluator('>', 'stl', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Blk':
                    correct.append(self.evaluator('>', 'blk', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Stl+Blk':
                    correct.append(self.evaluator('>', 'stl', 'blk', None, player_data, float(prediction[0][1:])))
            elif prediction[0][0] == 'u':
                if prediction[1] == 'Pts':
                    correct.append(self.evaluator('<', 'pts', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Ast':
                    correct.append(self.evaluator('<', 'ast', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Rebs':
                    correct.append(self.evaluator('<', 'trb', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == '3pt':
                    correct.append(self.evaluator('<', 'fg3', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Pts+Ast':
                    correct.append(self.evaluator('<', 'pts', 'ast', None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Pts+Rebs':
                    correct.append(self.evaluator('<', 'pts', 'trb', None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Rebs+Ast':
                    correct.append(self.evaluator('<', 'trb', 'ast', None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Pts+Rebs+Ast':
                    correct.append(self.evaluator('<', 'pts', 'ast', 'trb', player_data, float(prediction[0][1:])))
                if prediction[1] == 'Stl':
                    correct.append(self.evaluator('<', 'stl', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Blk':
                    correct.append(self.evaluator('<', 'blk', None, None, player_data, float(prediction[0][1:])))
                if prediction[1] == 'Stl+Blk':
                    correct.append(self.evaluator('<', 'stl', 'blk', None, player_data, float(prediction[0][1:])))
        predictions['Correct'] = correct
        return predictions.loc[predictions['Correct'] != 'X']
    
    def optimized_predictions_evaluator(self, optimized_predictions, current_evaluation):
        #return evaluations of past optimized predictions
        optimized_correct = []
        for i in range(len(optimized_predictions)):
            optimized_prediction = optimized_predictions.loc[i]
            if len(current_evaluation[(current_evaluation['Play'] == optimized_prediction['Play']) & (current_evaluation['Expert'] == optimized_prediction['Expert'])  & (current_evaluation['Odds'] == optimized_prediction['Odds'])]) == 0:
                optimized_correct.append('X')
            else:
                optimized_correct.append(current_evaluation[(current_evaluation['Play'] == optimized_prediction['Play']) & (current_evaluation['Expert'] == optimized_prediction['Expert'])  & (current_evaluation['Odds'] == optimized_prediction['Odds'])]['Correct'].values[0])
        return optimized_correct

    def all_evaluations(self):
        #return cumulative evaluations
        all_evals = pd.DataFrame()
        for i in os.listdir('/work/NBA-Bets-Evaluations/'):
            eval_ = pd.read_csv('/work/NBA-Bets-Evaluations/' + i)
            all_evals = all_evals.append(eval_)
        return all_evals

    def all_optimizer_evaluations(self):
        #return cumulative evaluations of optimized predictions
        #Optimized predictions began from December 1st to give 2 months worth of season training
        #Gains are defined as net units won per bet placed (e.g. gain of 0.5 for 10 bets would mean profit of 5) 
        all_optimized = pd.DataFrame()
        all_evals_december_onwards = pd.DataFrame()
        for i in os.listdir('/work/NBA-Bets-Optimized-Evaluations/'):
            all_optimized = all_optimized.append(pd.read_csv('/work/NBA-Bets-Optimized-Evaluations/' + i))
            all_evals_december_onwards = all_evals_december_onwards.append(pd.read_csv('/work/NBA-Bets-Evaluations/' + 'NBA-Bets-Evaluations-' + str(i).split('Optimized-Evaluations-')[1].split('.csv')[0] + ".csv"))
        optimized_gains = (sum(all_optimized[all_optimized['Correct'] == 'Y']['Payout']) - sum(all_optimized[all_optimized['Correct'] == 'N']['Units'])) / len(all_optimized)
        all_gains_december_onwards = (sum(all_evals_december_onwards[all_evals_december_onwards['Correct'] == 'Y']['Payout']) - sum(all_evals_december_onwards[all_evals_december_onwards['Correct'] == 'N']['Units'])) / len(all_evals_december_onwards)
        print('Optimized gains are ' + str(optimized_gains) + ' while regular bets gains are ' + str(all_gains_december_onwards))
        return