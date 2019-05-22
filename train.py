import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as T
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,classification_report
from neural_nets import DNN, MLP

class Train():
    def __init__(self, config):
        self.config = config

    def split_xy(self, data):
        cols = list(data.columns)
        cols.remove('No-show')
        self.features = cols
        X = data[cols]
        y = data['No-show']
        return np.asarray(X), np.asarray(y)

    def plot_featimpt(self, model):
        fi_dict = {}
        for f, imp in zip(self.features, model.feature_importances_):
            fi_dict[f] = imp

        fig, ax = plt.subplots(figsize=(15,10))
        ax.barh(range(len(fi_dict)), list(fi_dict.values()), align='center', color='lightgreen')
        ax.set_yticks(range(len(fi_dict)))
        ax.set_yticklabels(list(fi_dict.keys()))
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Feature Importance')
        ax.set_title('Tree Feature Importances')
        plt.savefig('./out/feature_impt.png')
        plt.close()

    def decision_tree(self, train, test):
        print('STARTING DECISION TREE')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        model = DecisionTreeClassifier(max_depth=self.config.max_depth,
                                       min_samples_split=self.config.min_samples_split,
                                       min_samples_leaf=self.config.min_samples_leaf,
                                       max_features=self.config.max_features)
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        self.plot_featimpt(model)

        return model

    def random_forest(self, train, test):
        print('STARTING RANDOM FOREST')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        model = RandomForestClassifier(n_estimators=self.config.n_estimators,
                                       max_depth=self.config.max_depth,
                                       min_samples_split=self.config.min_samples_split,
                                       min_samples_leaf=self.config.min_samples_leaf,
                                       max_features=self.config.max_features)
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        self.plot_featimpt(model)

        return model

    def extra_trees(self, train, test):
        print('STARTING EXTRA TREES CLASSIFIER')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        model = ExtraTreesClassifier(n_estimators=self.config.n_estimators,
                                       max_depth=self.config.max_depth,
                                       min_samples_split=self.config.min_samples_split,
                                       min_samples_leaf=self.config.min_samples_leaf,
                                       max_features=self.config.max_features)
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        self.plot_featimpt(model)

        return model

    def logistic_regression(self, train, test) :
        print('STARTING LOGISTIC REGRESSION')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        model = LogisticRegression(
            penalty='l2',
            max_iter=200
        )
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        return model

    def K_nearneighbor(self, train, test):
        print('STARTING KNN')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        model = KNeighborsClassifier(n_neighbors=self.config.neighbors)
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        return model

    def svm(self, train, test):
        print('STARTING SVM')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        model = LinearSVC(
            penalty='l2',
            C=self.config.C,
        )
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        return model

    def boosting(self, train, test, ada=False):
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)
        if ada :
            print('STARTING ADA BOOSTING')
            loss = 'exponential'
        else :
            print('STARTING GRADIENT BOOSTING')
            loss = 'deviance'
        model = GradientBoostingClassifier(
            n_estimators=200,
            loss = loss
        )
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        return model

    def bagging(self, train, test):
        print("STARTING BAGGING")
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)
        model = BaggingClassifier(
            n_estimators=50,
            bootstrap=True,
            bootstrap_features=True
        )
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        return model


    def neural_net(self, train, test, type):
        print('STARTING DEEP NEURAL NETWORKS')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        scaler = MinMaxScaler()
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)
        trainX = T.Tensor(trainX)
        testX = T.Tensor(testX)
        trainy = T.Tensor(trainy).view(trainy.shape[0],-1)
        testy = T.Tensor(testy).view(testy.shape[0],-1)
        if self.config.output_activation == 'softmax' :
            trainy = T.Tensor(trainy).view(-1,trainy.shape[0]).squeeze().long()
            testy = T.Tensor(testy).view(-1,testy.shape[0]).squeeze().long()
        if type == 'dnn' :
            model = DNN(self.config)
        elif type == 'mlp' :
            model = MLP(self.config)
        best_model = None
        losses = []
        test_losses = []
        agg_losses=[]
        for e in range(self.config.max_epoch):
            model.optimizer.zero_grad()
            train_pred = model(trainX)
            test_pred = model(testX)
            loss = model.loss(train_pred, trainy)
            test_loss = model.loss(test_pred, testy)
            losses.append(loss.item())
            test_losses.append(test_loss.item())
            agg_losses.append(loss.item() + test_loss.item())
            if e%10 == 0 :
                print('epoch', str(e + 1), ' - train loss : ', str(loss.item()), ' / - test loss : ', str(test_loss.item()))
            loss.backward()
            model.optimizer.step()

            # save best model
            if min(test_losses) >= test_loss.item() :
                best_model = model
                best_epoch = e+1

            # early stop
            if e > 30 :
                if np.std(losses[-10:]) < self.config.early_stop_std :
                    break
        print('\nThe best epoch was epoch {}\n'.format(str(best_epoch)))
        if self.config.output_activation == 'softmax' :
            train_pred = np.round(np.max(best_model(trainX).detach().numpy(),axis=1).flatten())
            test_pred = np.round(np.max(best_model(testX).detach().numpy(),axis=1).flatten())
        else :
            train_pred = np.round(best_model(trainX).detach().numpy().flatten())
            test_pred = np.round(best_model(testX).detach().numpy().flatten())
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        plt.plot(losses)
        plt.plot(test_losses)
        plt.title('Neural Network Cost Plot')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'])
        plt.savefig('./out/nn_train.png')
        plt.close()

        return best_model

    def multinomial_nb(self, train, test):
        print('STARTING MULTINOMIAL NAIVE BAYES')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        scaler = MinMaxScaler()
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)

        model = MultinomialNB()
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        return model

    def complement_nb(self, train, test):
        print('STARTING MULTINOMIAL NAIVE BAYES (COMPLEMENT)')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        scaler = MinMaxScaler()
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)

        model = ComplementNB()
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        return model

    def gaussian_nb(self, train, test):
        print('STARTING GAUSSIAN NAIVE BAYES')
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)

        scaler = MinMaxScaler()
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)

        model = GaussianNB()
        model.fit(trainX, trainy)
        train_pred = model.predict(trainX)
        test_pred = model.predict(testX)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

        return model

    def ensemble(self, models, models_oh, models_torch, train, test, train_oh, test_oh) :
        trainX, trainy = self.split_xy(train)
        testX, testy = self.split_xy(test)
        train_ohX, train_ohy = self.split_xy(train_oh)
        test_ohX, test_ohy = self.split_xy(test_oh)

        scaler = MinMaxScaler()
        scaler.fit(train_ohX)
        train_ohX = scaler.transform(train_ohX)
        test_ohX = scaler.transform(test_ohX)
        train_ohX = T.Tensor(train_ohX)
        test_ohX = T.Tensor(test_ohX)

        train_predictions_proba = []
        test_predictions_proba = []
        train_predictions = []
        test_predictions = []
        for model in models :
            train_pred_proba = model.predict_proba(trainX)
            train_pred_proba = np.max(train_pred_proba, axis = 1)
            train_predictions_proba.append(train_pred_proba)
            test_pred_proba = model.predict_proba(testX)
            test_pred_proba = np.max(test_pred_proba, axis = 1)
            test_predictions_proba.append(test_pred_proba)

            train_pred = model.predict(trainX)
            test_pred = model.predict(testX)
            train_predictions.append(train_pred)
            test_predictions.append(test_pred)
        for model in models_oh :
            train_pred_proba = model.predict_proba(train_ohX)
            train_pred_proba = np.max(train_pred_proba, axis = 1)
            train_predictions_proba.append(train_pred_proba)
            test_pred_proba = model.predict_proba(test_ohX)
            test_pred_proba = np.max(test_pred_proba, axis = 1)
            test_predictions_proba.append(test_pred_proba)

            train_pred = model.predict(train_ohX)
            test_pred = model.predict(test_ohX)
            train_predictions.append(train_pred)
            test_predictions.append(test_pred)
        for model in models_torch :
            train_pred_proba = model(train_ohX).detach().numpy().flatten()
            train_predictions_proba.append(train_pred_proba)
            test_pred_proba = model(test_ohX).detach().numpy().flatten()
            test_predictions_proba.append(test_pred_proba)

            train_pred = np.round(train_pred_proba)
            test_pred = np.round(test_pred_proba)
            train_predictions.append(train_pred)
            test_predictions.append(test_pred)


        # ensemble the predictions
        print('\nENSEMBLING THE RESULTS...')
        if self.config.ensemble_method == 'vote' :
            train_pred = np.round(np.mean(train_predictions,axis=0)).transpose()
            test_pred = np.round(np.mean(test_predictions,axis=0)).transpose()
        elif self.config.ensemble_method == 'mlp' :
            ens_model = MLPClassifier(
                activation='logistic',
                hidden_layer_sizes=(8,4),
                alpha=0.1,
                max_iter=300,
            )
            tr = np.array(train_predictions_proba).transpose()
            te = np.array(test_predictions_proba).transpose()
            ens_model.fit(tr, trainy)
            train_pred = ens_model.predict(tr)
            test_pred = ens_model.predict(te)
        cm = confusion_matrix(trainy, train_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('------train evaluation------')
        print(cm)
        print(classification_report(trainy, train_pred))
        print('TRAIN ACCURACY : {}\n'.format(np.round(acc,4)))
        cm = confusion_matrix(testy, test_pred)
        acc = (cm[0][0]+cm[1][1])/(np.sum(cm))
        print('\n------test evaluation------')
        print(cm)
        print(classification_report(testy, test_pred))
        print('TEST ACCURACY : {}\n\n'.format(np.round(acc,4)))

