#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:33:56 2017

@author: kevinwu
"""
# Training the ANN with Stochastic Gradient Descent 
# First layer - input layer 
# Step 1: Randomly initialise the weights to small numbers close to 0 (but not 0)
# Step 2: Input the first observation of the dataset in the input layer, each feature in one input node (11 input nodes)
# Step 3: Activation function hidden layer (rectifier), sigmoid (good for output layer)
# Step 4: Compare the predicted result o the actual result, measure the generated error
# Step 5: Back-propagation - learning rate decideds how much we update the weights
# Step 6: update the weights after each observation
# Step 7: Train

# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encode before splitting because matrix X and independent variable Y must be already encoded
# Found two categorical data (country, gender)
# create dummy variables, avoid dummy variable trap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# lots of high computation to ease calculation, we don't want one independent variable dominating
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN

# Importing the Keras libraries and package
# Sequential module - initialize neural network
# Dense - layers of ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
# Take average of input + output for units/output_dim param in Dense
# input_dim is necessary for the first layer as it was just initialized
classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer with dropout
# doesn't need the input_dim params
# kernel_initializer updates weights
# activation function - rectifier
classifier.add(Dense(6, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
# dependent variable with more than two categories (3), output_dim needs to change (e.g. 3), activation function - sufmax
classifier.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid' ))

# Compiling the ANN - applying Stochastic Gradient Descent to whole ANN
# Several different SGD algorithms
# mathematical details based on the loss function
# binary_crossentropy, categorical_cross_entropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set
# X_train, y_train, Batch size, Epochs (whole training set)
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
# Training set, see if the new data probability is right
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Imrpvoing and Tuning the ANN

# Evaluating the ANN 
# Keras wrapper and Sci-kit Learn for k-Fold Cross Validation 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(6, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid' ))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# k-Fold cross validator to check if the real relevant accuracy or the second one 
#ß and where we are in bias-variance tradeoffs
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variable = accuracies.std()

# overfitting is when it's trained to much on the training set, less performant, test set
# and training set, high variance in, Dropout Regulariatio to reduce overfitting if needed

# Tuning the ANN, parameters learned during training (weights), stay fixed (hyperparamers - fixed, epochs, neurons)
# parameter tuning best value of these hyperparameters, GridSearchCV with k-Fold Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimzer):
    classifier = Sequential()
    classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(6, kernel_initializer = 'glorot_uniform', activation = 'relu' ))
    classifier.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid' ))
    classifier.compile(optimizer = optimzer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimzer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy', s
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_