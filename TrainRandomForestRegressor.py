from sklearn.model_selection import RandomizedSearchCV,  train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Temp
import pandas as pd
import numpy as np
import os


class TrainRandomForestRegressor:
    """"""
    def __init__(self, features, labels, hyperparameterTunning = True, randomGrid = "Default", \
        preprocessing = False, scoring = "neg_mean_squared_error", n_iter =10, cv = 5, verbose = 2):
        
        self.__model = RandomForestRegressor()
        self.__hyperparameterTunning = hyperparameterTunning
        # Constructing random Grid:
        if hyperparameterTunning:
            if randomGrid == "Default":
                # Number of trees in random forest
                n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
                # Number of features to consider at every split
                max_features = ['auto', 'sqrt']
                # Maximum number of levels in tree
                max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
                # Minimum number of samples required to split a node
                min_samples_split = [2, 5, 10, 15, 100]
                # Minimum number of samples required at each leaf node
                min_samples_leaf = [1, 2, 5, 10]
                
                # Create Dictionary for Random Grid
                self.__randomGrid = { 'n_estimators': n_estimators,
                                        'max_features': max_features,
                                        'max_depth': max_depth,
                                        'min_samples_split': min_samples_split,
                                        'min_samples_leaf': min_samples_leaf}
                self.n_iter = n_iter
                self.cv = cv
                self.verbose = verbose
                self.scoring = scoring

            else:
                self.__randomGrid = randomGrid
        
        self.__splitDataset(features,labels)
        self.train()
        self.evaluateModel()

    def __splitDataset(self, features, labels, testSize = 0.2):
        self.__featuresTrain, self.__featuresTest, self.__labelsTrain, \
            self.__labelsTest = train_test_split(features, labels, test_size = testSize)

    def train(self):
        if self.__hyperparameterTunning:
            self.__trainer = RandomizedSearchCV(estimator = self.__model, param_distributions = self.__randomGrid, 
                               scoring = self.scoring, n_iter = self.n_iter, cv = self.cv, 
                               verbose=self.verbose, random_state=42, n_jobs = 1)

        else:
            self.__trainer = self.__model
        self.__trainer.fit(self.__featuresTrain, self.__labelsTrain)

    def getBestParams(self):
        return self.__trainer.best_params_

    def evaluateModel(self):
        predictedLabelsTrain = self.predict(self.__featuresTrain)
        predictedLabelsTest = self.predict(self.__featuresTest)
        trainMSE = mean_squared_error(self.__labelsTrain, predictedLabelsTrain )
        trainRMSE = mean_squared_error(self.__labelsTrain, predictedLabelsTrain, squared=False )
        trainMAE = mean_absolute_error(self.__labelsTrain, predictedLabelsTrain )
        testMSE = mean_squared_error(self.__labelsTest, predictedLabelsTest )
        testRMSE = mean_squared_error(self.__labelsTest, predictedLabelsTest, squared=False )
        testMAE = mean_absolute_error(self.__labelsTest, predictedLabelsTest )
        print("="*100)
        print("Evaluating Model for training dataset: ")
        print("MSE  :", trainMSE)
        print("RMSE :", trainRMSE)
        print("MAE  :", trainMAE)
        print("="*100)
        print("Evaluating Model for Tesing dataset: ")
        print("MSE  :", testMSE)
        print("RMSE :", testRMSE)
        print("MAE  :", testMAE)
        
    def predict(self,features):
        predictedLabel = self.__trainer.predict(features)
        return predictedLabel



