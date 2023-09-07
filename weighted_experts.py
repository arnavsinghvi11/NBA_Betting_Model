import warnings
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import sys
import xgboost as xgb
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if os.name == 'nt':  # for Windows OS
    os.devnull = 'nul'
all_box_score_results = pd.read_csv("All_Box_Score_Results_updated.csv")
all_box_score_results = all_box_score_results[all_box_score_results['mp'] > 1]
all_box_score_results.insert(loc=0, column='player_id', value = all_box_score_results.set_index(['name','team']).index.factorize()[0]+1)
all_box_score_results = all_box_score_results.sort_values(by = ['date', 'player_id'], ascending = [True, True])

def last_x_avgs(data, prop_type, features_set, x):
    last_x_games = data[-x:]
    last_x_stats = last_x_games[features_set].mean(axis=0).to_frame().T
    last_x_stats = last_x_stats.drop(prop_type, axis=1)
    return last_x_stats

def preprocess_model(past_data, prop_type, features_set):
    info_data = past_data[features_set]
    info_data = info_data.dropna()
    features = list(info_data.columns)
    features.remove(prop_type.lower())
    target = [prop_type.lower()]
    X = info_data.loc[:, features].values
    y = info_data.loc[:, target].values
    return train_test_split(X, y, test_size=0.1, random_state=1)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def models(X_train, y_train):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))
    model_defs = {
        'DNN': Sequential([
            normalizer,
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1)
        ]),
        'MLP': MLPRegressor(hidden_layer_sizes=(100,), solver='adam', random_state=42, max_iter=5000),
        'RF': RandomForestRegressor(n_estimators=200, random_state=42),
        'XGB': XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, 
                            max_depth=8, subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0),
        'LSTM': Sequential([
            LSTM(128, input_shape=(X_train.shape[1], 1)),
            Dense(1)
        ])
    }
    model_defs['DNN'].compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model_defs['LSTM'].compile(optimizer='adam', loss='mse')
    threads = {name: threading.Thread(target=train_model, args=(model, X_train, y_train)) for name, model in model_defs.items()}
    [thread.start() for thread in threads.values()]
    [thread.join() for thread in threads.values()]
    return model_defs

def predict_model(model, player_last_avgs, player_preds):
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        player_pred = model.predict(player_last_avgs)
        sys.stdout = sys.__stdout__
        player_preds.append(player_pred)

def models_eval(past_data, past_x_games, prop_type, features_set, trained_models):
    player_data_list = []
    player_preds = {name: [] for name in trained_models}
    for i in past_data['player_id'].unique():
        player_data = past_data[past_data['player_id'] == i].fillna(0)
        player_last_avgs = last_x_avgs(player_data, prop_type.lower(), features_set, past_x_games)
        player_data_list.append({
            'PlayerName': player_data['name'].values[0],
            'PlayerTeam': player_data['team'].values[0],
            'PlayerLastAvgs': player_last_avgs
        })
    for pdata in player_data_list:
        threads = {model_name: threading.Thread(target=predict_model, args=(model, pdata['PlayerLastAvgs'], player_preds[model_name])) 
                   for model_name, model in trained_models.items()}
        [thread.start() for thread in threads.values()]
        [thread.join() for thread in threads.values()]
    predictions = pd.DataFrame({
        'PlayerName': [pdata['PlayerName'] for pdata in player_data_list],
        'PlayerTeam': [pdata['PlayerTeam'] for pdata in player_data_list],
        **{f'PlayerPredictions_{name}_{prop_type}': preds for name, preds in player_preds.items()}
    })
    return predictions

def get_model_names_from_predictions(predictions, prop_type):
    return [col.split('_')[1] for col in predictions.columns if f"PlayerPredictions_" in col and prop_type in col]

def get_prediction(predictions, model_name, prop_type, player_name, team):
    return float(predictions[(predictions['PlayerName'] == player_name) & (predictions['PlayerTeam'] == team)][f'PlayerPredictions_{model_name}_{prop_type}'].values[0][0])

def models_predictions_eval(folder_path, predictions, optimized, unoptimized, prop_type):
    model_names = get_model_names_from_predictions(predictions, prop_type)
    preds_dict = {name: [] for name in model_names}
    optimizer_preds = []
    for play, team in zip(unoptimized['Play'].values, unoptimized['Teams'].values):
        player_name = play.split(' ')[0]
        bet_value = float(play.split(' ')[1][1:])
        over_under = play.split(' ')[1][0]
        if play not in optimized['Play'].values:
            optimizer_preds.append(0)
        else:
            optimizer_preds.append(1)
        for model_name in model_names:
            prediction = get_prediction(predictions, model_name, prop_type, player_name, team)
            is_over = int(prediction > bet_value)
            is_under = int(prediction < bet_value)
            preds_dict[model_name].append(is_over if over_under == 'o' else is_under)
    prediction_path = os.path.join(folder_path, 'predictions')
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    predictions_df_dict = {}
    for model_name in model_names:
        predictions_df_dict[model_name] = unoptimized[[bool(x) for x in preds_dict[model_name]]]
        predictions_df_dict[model_name].to_csv(os.path.join(prediction_path, f'{model_name}_{prop_type}.csv'))
    all_preds = [optimizer_preds] + list(preds_dict.values())
    all_predictions_dfs = [optimized] + list(predictions_df_dict.values())
    return all_preds, all_predictions_dfs

def weighted_optimizer(folder_path, expert_names, expert_weights, prop_type, all_predictions_dfs):
    expert_to_df_map = {expert_name: all_predictions_dfs[i] for i, expert_name in enumerate(expert_names)}
    indices = np.array(expert_weights).argsort()[-2:][::-1]
    best_experts = [expert_names[index] for index in indices]
    major1 = expert_to_df_map[best_experts[0]]
    major2 = expert_to_df_map[best_experts[1]]
    if not os.path.exists(folder_path + '/optimizations'):
        os.makedirs(folder_path + '/optimizations')
    try:
        int_columns = ['Unnamed: 0', 'Odds', 'Net Units Record', 'hmcrt_adv']
        float_columns = ['Units', 'Payout', 'Profit']
        major1[int_columns] = major1[int_columns].apply(pd.to_numeric, errors='coerce')
        major1[float_columns] = major1[float_columns].astype(float)
        major2[int_columns] = major2[int_columns].apply(pd.to_numeric, errors='coerce')
        major2[float_columns] = major2[float_columns].astype(float)
        intersection = pd.merge(major1, major2, how='inner').drop_duplicates()
        intersection.to_csv(folder_path + '/optimizations/' + 'intersection_' + str(prop_type) + '.csv')
        combined = pd.concat([major1, major2], axis=0).drop_duplicates()
        combined.to_csv(folder_path + '/optimizations/' + 'combined_' + str(prop_type) + '.csv')
        return intersection, combined
    except Exception as e:
        print('Error:', e)

def eval_past(folder_path, month, date_counter, prop_type, intersection, combined):
    past_eval = pd.read_csv('NBA-Bets-Evaluations/NBA-Bets-Evaluations-' + str(month) + '-' + str(date_counter) + '.csv')
    if prop_type == 'trb':
        prop_type = 'Rebs'
    if prop_type == 'fg3':
        prop_type = '3pt M'
    past_eval = past_eval.loc[past_eval['Play'].str.contains(prop_type, regex=False)]
    if '+' not in prop_type:
        past_eval = past_eval.loc[~past_eval['Play'].str.contains(r'\+')]
    if prop_type == 'Rebs+Ast':
        past_eval = past_eval.loc[~past_eval['Play'].str.contains('Pts')]
    elif prop_type == 'Pts+Rebs':
        past_eval = past_eval.loc[~past_eval['Play'].str.contains('Ast')]
    if not os.path.exists(folder_path + '/evals'):
        os.makedirs(folder_path + '/evals')
    int_columns = ['Unnamed: 0', 'Odds', 'Net Units Record', 'hmcrt_adv', 'All Game Correct', 'Last 5 Correct', 'Last 10 Correct']
    float_columns = ['Units', 'Payout', 'Profit', 'All Game Percentages', 'Last 5 Percentages', 'Last 10 Percentages']
    past_eval[int_columns] = past_eval[int_columns].apply(pd.to_numeric, errors='coerce')
    past_eval[float_columns] = past_eval[float_columns].astype(float)
    intersection_eval = pd.merge(intersection, past_eval)
    intersection_eval.to_csv(folder_path + '/evals/' + 'intersection_eval_' + str(prop_type) + '.csv')
    float_columns = ['Units', 'Payout', 'Profit', 'All Game Percentages', 'Last 5 Percentages', 'Last 10 Percentages', 'All Game Correct', 'Last 5 Correct', 'Last 10 Correct']
    past_eval[float_columns] = past_eval[float_columns].astype(float)
    combined_eval = pd.merge(combined, past_eval)
    combined_eval.to_csv(folder_path + '/evals/' + 'combined_eval_' + str(prop_type) + '.csv')
    return [1 if val == 'Y' else 0 for val in past_eval['Correct'].values]

def weighted_experts(expert_weights, expert_history, cost_threshold, num_experts, window_size, eta, all_preds, true_labels):
    decay_rate = 0.9
    for i in range(len(true_labels)):
        true_label = true_labels[i]
        expert_predictions = [preds[i] for preds in all_preds]
        weighted_predictions = expert_weights * expert_predictions
        total_weighted_prediction = np.sum(weighted_predictions)
        final_prediction = 1 if total_weighted_prediction > cost_threshold else 0
        for j in range(num_experts):
            expert_error = abs(expert_predictions[j] - true_label)
            expert_history[j, i % window_size] = expert_error
            decayed_eta = eta * (decay_rate ** i)
            expert_weights[j] *= np.exp(-decayed_eta * expert_error)
        expert_weights /= np.sum(expert_weights)
    return expert_weights

def get_past_data(past_data, date_counter2):
    if date_counter2 not in past_data['date'].values:
        flag = True
        temp_date_counter2 = date_counter2
        while flag:
            str_number = str(temp_date_counter2)
            if str_number.endswith("10"):
                str_number = str_number[:-3] + str_number[-2:]
            temp_date_counter2 = int(str_number)
            temp_date_counter2 -= 1
            if temp_date_counter2 not in past_data['date'].values:
                flag = True
            else:
                past_data = past_data.loc[:past_data.loc[past_data['date'] == temp_date_counter2].index[-1]]
                flag = False
    else:
        past_data = past_data.loc[:past_data.loc[past_data['date'] == date_counter2].index[-1]]
    return past_data

def process_csv(filename, prop_type):
    data = pd.read_csv(filename)
    data = data.loc[data['Play'].str.contains(prop_type, regex=False)]
    if '+' not in prop_type:
        data = data.loc[~data['Play'].str.contains(r'\+')]
    if prop_type == 'Rebs+Ast':
        data = data.loc[~data['Play'].str.contains('Pts')]
    elif prop_type == 'Pts+Rebs':
        data = data.loc[~data['Play'].str.contains('Ast')]
    return data

def prop_type_conversion(prop_type):
    conversions = {'Rebs': 'trb', '3pt M': 'fg3'}
    return conversions.get(prop_type, prop_type)

def prop_prediction_with_experts(folder_path, month, date_counter, date_counter2, prop_type, features_set, past_data, past_x_games, expert_names, expert_weights_diff, expert_history_diff, cost_threshold, num_experts, window_size, eta, passed_in_data1=None):
    file_path = f'NBA-Bets-Predictions/NBA-Bets-{month}-{date_counter}.csv'
    unoptimized = process_csv(file_path, prop_type)
    if unoptimized.empty:
        print('Empty')
        if passed_in_data1 is not None:
            models_predictions = passed_in_data1
        else:
            prop_type = prop_type_conversion(prop_type)
            past_data = get_past_data(past_data, date_counter2)
            X_train, X_val, y_train, y_val = preprocess_model(past_data, prop_type, features_set)
            trained_models = models(X_train.astype(float), y_train.astype(float))
            models_predictions = models_eval(past_data, past_x_games, prop_type, features_set, trained_models)
    else:
        optimized = process_csv(f'NBA-Bets-Optimized-Predictions/NBA-Optimized-Bets-{month}-{date_counter}.csv', prop_type)
        prop_type = prop_type_conversion(prop_type)
        past_data = get_past_data(past_data, date_counter2)
        if '+' not in prop_type:
            X_train, X_val, y_train, y_val = preprocess_model(past_data, prop_type, features_set)
            trained_models = models(X_train.astype(float), y_train.astype(float))
            models_predictions = models_eval(past_data, past_x_games, prop_type, features_set, trained_models)
        else:
            models_predictions = passed_in_data1
        all_preds, all_predictions_dfs = models_predictions_eval(folder_path, models_predictions, optimized, unoptimized, prop_type)
        intersection, combined = weighted_optimizer(folder_path, expert_names, expert_weights_diff, prop_type, all_predictions_dfs)
        true_labels = eval_past(folder_path, month, date_counter, prop_type, intersection, combined)
        expert_weights_diff = weighted_experts(expert_weights_diff, expert_history_diff, cost_threshold, num_experts, window_size, eta, all_preds, true_labels)
    prop_type = prop_type.replace('\\', '')
    with open(os.path.join(folder_path, f'expert_weights_{prop_type}.pkl'), 'wb') as f:
        pickle.dump(expert_weights_diff, f)
    with open(os.path.join(folder_path, f'expert_history_{prop_type}.pkl'), 'wb') as f:
        pickle.dump(expert_history_diff, f)
    return expert_weights_diff, expert_history_diff, models_predictions

eta = 0.1
cost_threshold = 0.5
expert_names = ['optimizer_preds', 'mlp_preds', 'dnn_preds', 'xgb_preds', 'lstm_preds', 'rf_preds']
num_experts = len(expert_names)
window_size = 10
past_x_games = 10
directory = '/content/drive/MyDrive/all_10/' #directory to be changed
file_suffixes = ['Pts', 'Ast', 'trb', 'Pts+Ast', 'Pts+Rebs', 'Rebs+Ast', 'Pts+Rebs+Ast', 
                 'Stl', 'Blk', 'Stl+Blk', 'fg3']
stats_dict = {
    'Pts': ['pts', 'fg3', 'ts_pct', 'efg_pct', 'fg3a_per_fga_pct', 'usg_pct', 'off_rtg', 'bpm', 'mp'],
    'Ast': ['ast', 'ast_pct', 'usg_pct', 'off_rtg', 'bpm', 'mp'],
    'Rebs': ['trb', 'orb_pct', 'drb_pct', 'trb_pct', 'usg_pct', 'off_rtg', 'bpm', 'mp'],
    'Stl': ['stl', 'stl_pct', 'def_rtg', 'bpm', 'mp'],
    'Blk': ['blk', 'blk_pct', 'def_rtg', 'bpm', 'mp'],
    '3pt M': ['fg3', 'pts', 'ts_pct', 'efg_pct', 'fg3a_per_fga_pct', 'usg_pct', 'off_rtg', 'bpm', 'mp']
}
combo_stats = ['Pts+Ast', 'Pts+Rebs', 'Rebs+Ast', 'Pts+Rebs+Ast', 'Stl+Blk']
models = ["DNN", "MLP", "RF", "XGB", "LSTM"]

def load_files(directory, suffixes, prefix):
    data = {}
    for suffix in suffixes:
        with open(f'{directory}{prefix}{suffix}.pkl', 'rb') as f:
            data[suffix] = pickle.load(f)
    return data

input_value = input("Enter 'start' or 'middle':")
if input_value == 'start':
    if not os.path.exists(directory):
        os.makedirs(directory)
    expert_weights, expert_history = {}, {}
    for suffix in file_suffixes:
        expert_weights[suffix] = np.ones(num_experts)
        expert_history[suffix] = np.zeros((num_experts, window_size))
else:
    input_month = str(input("Enter month:"))
    input_day = str(input("Enter day"))
    directory_base = f'{directory}{input_month}/{input_day}/'
    expert_weights = load_files(directory_base, file_suffixes, 'expert_weights_')
    expert_history = load_files(directory_base, file_suffixes, 'expert_history_')

start_date = datetime.date(2022, 11, 1)
end_date = datetime.date(2023, 6, 30)
date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

month_ends = {}
for date in date_range:
    month = date.month
    last_day = date.strftime('%d')
    if month < 10:
        month_ends[month] = int(str(month) + ('0' * (4 + (month - 1))) + str(int(last_day)))
    else:
        month_ends[month] = int(str(month) + last_day)
month_order = [12, 1, 2, 3, 4, 5, 6]
if input_value == 'middle':
    date_counter = int(input_day) + 1
    month_order = month_order[month_order.index(int(input_month)):]
else:
    date_counter = 1
for month in month_order:
    last_day = 31
    if month == 2:
        last_day = 28
    elif month == 4 or month == 6:
        last_day = 30
    while date_counter <= last_day:
        if date_counter == 1:
            if month == 12:
                date_counter2 = month_ends[11]
            elif month == 1:
                date_counter2 = month_ends[12]
            else:
                date_counter2 = month_ends[month - 1]
        else:
            if date_counter - 1 < 10:
                if month == 12:
                    date_counter2 = int(str(120) + str(date_counter - 1))
                else:
                    if date_counter - 1 == 9:
                        date_counter2 = int(str(month) + ('0' * (5 + (month - 1))) + str(date_counter - 1))
                    else:
                        date_counter2 = int(str(month) + ('0' * (4 + (month - 1))) + str(date_counter - 1))
            else:
                if month == 12:
                    date_counter2 = int(str(12) + str(date_counter - 1))
                else:
                    date_counter2 = int(str(month) + ('0' * (4 + (month - 1))) + str(date_counter - 1))
        if (date_counter2+1) not in all_box_score_results['date'].unique() and date_counter2 != 1130 and date_counter2 != 1231 and date_counter2 != 1000031 and date_counter2 != 20000028 and date_counter2 != 300000031 and date_counter2 != 4000000030 and date_counter2 != 50000000031 and date_counter2 != 60000000030:
            models_predictions_pts, models_predictions_ast, models_predictions_rebs, models_predictions_stl, models_predictions_blk, models_predictions_3pt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            if month != 12:
                 if date_counter == 10:
                       date_counter2 = int(str(month) + ('0' * (4 + (month - 1))) + str(date_counter - 1))
            month_folder = os.path.join(directory, str(month))
            if not os.path.exists(month_folder):
                os.makedirs(month_folder)
            day_folder = os.path.join(month_folder, str(date_counter))
            if not os.path.exists(day_folder):
                os.makedirs(day_folder)
            model_predictions_dict = {}

            for stat, attributes in stats_dict.items():
                expert_weights[stat], expert_history[stat], model_predictions_dict[stat] = prop_prediction_with_experts(
                    day_folder, month, date_counter, date_counter2, stat, attributes, all_box_score_results, past_x_games, expert_names, 
                    expert_weights[stat], expert_history[stat], cost_threshold, num_experts, window_size, eta
                )
        for combo in combo_stats:
            stats = [stat if stat != 'Rebs' else 'trb' for stat in combo.split('+')]
            if all([not model_predictions_dict[stat].empty for stat in stats]):
                merged_predictions = model_predictions_dict[stats[0]]
                for stat in stats[1:]:
                    merged_predictions = pd.merge(merged_predictions, model_predictions_dict[stat], on=['PlayerName', 'PlayerTeam'], how='inner')
                for model in models:
                    merged_predictions[f'PlayerPredictions_{model}_{combo}'] = sum([merged_predictions[f'PlayerPredictions_{model}_{stat}'] for stat in stats])
                expert_weights[combo], expert_history[combo], model_predictions_dict[combo] = prop_prediction_with_experts(
                    day_folder, month, date_counter, date_counter2, combo, [], all_box_score_results, past_x_games, expert_names, expert_weights[combo], 
                    expert_history[combo], cost_threshold, num_experts, window_size, eta, merged_predictions
                )
        date_counter += 1
    date_counter = 1