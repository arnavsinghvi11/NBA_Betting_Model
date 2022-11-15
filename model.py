import pandas as pd
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class Model:
    def convert_dummies(self, all_evals, column, prefix):
        #returns one-hot encoding of categorical variables
        dummy = pd.get_dummies(all_evals[column], prefix=prefix)
        all_evals = all_evals.drop(columns = [column])
        return pd.concat([all_evals.reset_index(drop=True),dummy.reset_index(drop=True)], axis=1)

    def features_remover(self, feature, features_list):
        #removes unneeded features from inputted features 
        if feature in features_list:
            features_list.remove(feature)
        return features_list

    def over(self, row):
        #returns 1 if prop bet is type - over and 0 if prop bet is type - under 
        plays = [i for i in row['Play'].split(' ') if i != '']
        if plays[1][0] == 'o' or plays[1] == 'Yes':
            val = 1
        else:
            val = 0
        return val

    def prop_bet_finder(self, row):
        #returns prop bet by extracting statistic type
        plays = [i for i in row['Play'].split(' ') if i != ''][2:]
        if len(plays) > 1:
            if plays[1] == '+':
                return plays[0]
            return plays[0] + plays[1]
        return plays[0]

    def preprocessing(self, data_input, is_evaluation):
        #conducts feature engineering over bets inputs and outputs preprocessed input
        data_input['Over?'] = data_input.apply(self.over, axis = 1)
        data_input['Under?'] = 1 - data_input['Over?']
        data_input['Prop Bets'] = data_input.apply(self.prop_bet_finder, axis = 1) 
        data_input = self.convert_dummies(data_input, 'Prop Bets', 'Prop_Bet_Type_')
        data_input = self.convert_dummies(data_input, 'Expert', 'Expert_')
        data_input = self.convert_dummies(data_input, 'Teams', 'Team_')
        data_input = self.convert_dummies(data_input, 'opponent', 'Opp_')
        data_input = data_input.drop(columns = ['Play'])
        if is_evaluation:
            correct = data_input.pop('Correct')
            data_input['Correct'] = correct
        return data_input
    
    def features_processing(self, features):
        #removes extraneous features from features set
        features = self.features_remover('name', features)
        features = self.features_remover('Profit', features)
        features = self.features_remover('Net Units Record', features)
        features = self.features_remover('Odds', features)
        features = self.features_remover('Units', features)
        features = self.features_remover('Payout', features)
        return features

    def train(self, features, target, data_input):
        #model training to predict if bets prediction will come true
        X = data_input.loc[:, features].values
        Y = data_input.loc[:, target].values

        #split into training and testing sets 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X_train)
        test_scaled = scaler.transform(X_test)

        models = []
        models.append(('LR', LogisticRegression(random_state=0, C= 0.4, solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 0.91)))
        models.append(('KNN', KNeighborsClassifier(leaf_size = 1, p = 1, n_neighbors = 5)))
        models.append(('NB', GaussianNB(var_smoothing = 1.0)))
        models.append(('SVM', SVC(kernel='linear', C=1, gamma = 0.001, probability=True)))
        models.append(('MLP', MLPClassifier(random_state = 0)))
        models.append(('BAG', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state = 0)))
        models.append(('RFC', RandomForestClassifier(n_estimators=200, random_state = 0)))
        models.append(('EX', ExtraTreesClassifier(n_estimators=200,  random_state = 0)))
        models.append(('ADA', AdaBoostClassifier(n_estimators=100,  random_state = 0)))
        models.append(('STO', GradientBoostingClassifier(n_estimators=100,  random_state = 0)))

        # evaluate each model in turn
        predictions, top5_models, final_models = [], [], []
        for name, model in models:
            model.fit(train_scaled, y_train)
            print(name)
            print(model.score(test_scaled, y_test))
            top5_models.append([model, model.score(test_scaled, y_test)])
        top5_models.sort(key = lambda x: x[1], reverse = True)
        top5_models = top5_models[:5]
        for model1 in top5_models:
            for model2 in models:
                if model1[0] == model2[1]:
                    final_models.append(model2)
        ensemble = VotingClassifier(final_models)
        return X, Y, ensemble