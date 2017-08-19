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

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Take average of input + output for units/output_dim param in Dense
classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'glorot_uniform', activation = 'relu' ))

# Part 3 - Making the predictions and evaluating the model

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)