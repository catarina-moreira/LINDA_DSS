# update for tensorflow
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import seaborn as sns
import re
import warnings
import csv

# keras / deep learning libraries
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

# callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import pylab as pl
from pylab import savefig
plt.style.use('seaborn-deep')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler


# RECALL -----------------------------------------------------------------------------
# Computes the recal measure of an evaluation setting
# y_true: list of groundtruth labels
# y_pred: list of predictions from blackbox
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

# PRECISION ---------------------------------------------------------------------------
# Computes the precision measure of an evaluation setting
# y_true: list of groundtruth labels
# y_pred: list of predictions from blackbox
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

# F1 ------------------------------------------------------------------------------------
# Computes the F1 measure of an evaluation setting
# y_true: list of groundtruth labels
# y_pred: list of predictions from blackbox
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# CREATE_MODEL --------------------------------------------------------------------------
# creates a neural network model with a certain number of hidden layers and a certain 
# number of neurons in each layer.
# input_dim: an integer specifying the number of input neurons
# output_dim: an integer specifying the number of output neurons (the number of labels)
# hidden_layers: an integer specifying the number of hidden layers
# loss_func: the loss function of the model. By default, it is applied the 'categorical_crossentropy'
# optim: the optimisation algorithm used in the model. By default it is used the 'nadam' algorithm
# metrics: a list of strings specifying the metrics to be evaluated ('accuracy', 'f1', 'recall','precision')
def create_model(input_dim, output_dim, nodes, hidden_layers=1, loss_func='categorical_crossentropy', optim='nadam', metrics=['accuracy'], name='model'):
    
    model = Sequential(name=name)
    model.add( Dense(nodes, input_dim=input_dim, activation='relu'))  # input layer
    for i in range(hidden_layers):                                    # hidden layers
        model.add(Dense(nodes, activation='relu'))  
    model.add(Dense(output_dim, activation='softmax'))                # output layer

    if( optim == "nadam" ):                                           # Compile model
        optim = keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    model.compile(loss=loss_func, optimizer=optim, 
                  metrics=metrics)
    return model
    
# GRID_SEARCH -----------------------------------------------------------------------------
# Generates a set of models with different configurations, ranging from an
# initial number of neurons to a maximum number of neurons
# start_nodes: an integer specifying the initial number of neurons to generate a model from
# max_nodes:   an integer specifying the maximum number of neurons to generate a model from
# max_hlayers: an integer specifying the maximum number of hidden layers to generate a model from
# debug: boolean that acts as a flag. If True, it displays the characteristics of each model
# metrics: a list of strings with the metrics to be evaluated 
def grid_search_model_generator(n_features, n_classes, start_nodes = 1, max_nodes = 12, max_hlayers = 5, debug = False, metrics = ['accuracy'] ):

    models = []

    # generate different models with different neurons and different hidden layers
    for neurons in range(start_nodes, max_nodes+1):
        for hidden_layer in range(1, max_hlayers+1):
            model_name = "model_h" + str(hidden_layer) + "_N"+str(neurons)
            model = create_model(n_features, n_classes, neurons, hidden_layer, name=model_name, metrics = metrics)
            models.append( model )  # add the generated model to a list

    # plot general information for each model
    if( debug ):  
        for model in models:
            model.summary()

    return models

# PERFORM_GRID_SEARCH -------------------------------------------------------------------
# given a list of models with different configurations, fit the data to the models,
# and evaluate the model. This function returns a list of training histories for each model
# models: list of models
# X_train: 
# Y_train: 
# X_validation: 
# Y_validation: 
# X_test: 
# Y_test: 
# batch_size: 
# epochs: 
def perform_grid_search( models, path, dataset_name, X_train, Y_train, X_validation, Y_validation, X_test, Y_test, batch_size, epochs ):

	HISTORY_DICT = {}
	
	# define the callebacks to take into consideration during training
	# stop training when convergence is achieved after 10 iterations
	early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
	
	# save the best model after every epoch
	model_checkpoint = ModelCheckpoint(path + "training/" + dataset_name + "/model_{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss',  verbose=0, save_best_only=True,  mode='min')
	callbacks_list = [early_stop, model_checkpoint]
	
	# grid search over each model
	for model in models:
		
		print('MODEL NAME:', model.name)
		history_callback = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose=0, validation_data=(X_validation, Y_validation), callbacks=callbacks_list)
		score_test = model.evaluate( X_test, Y_test, verbose=0 )
		score_train = model.evaluate( X_train, Y_train  )
		
		print('Test loss:     ', format(score_test[0], '.4f'), '\tTrain loss: ', format(score_train[0], '.4f') )
		print('Test accuracy: ', format(score_test[1], '.4f'), '\tTrain accu: ', format(score_train[1], '.4f') )
		print('Abs accuracy:  ', format( np.abs( score_test[1] - score_train[1] ), '.4f'))
		print('Abs loss:      ', format( np.abs( score_test[0] - score_train[0] ), '.4f'))
		print('\n###########################################################\n')
		
		HISTORY_DICT[model.name] = [history_callback, model]
	
	return HISTORY_DICT

# SAVE_MODEL -----------------------------------------------------------------------------
# saves a trained model into a json and hdf5 file
# model: model to be saved
# model_name: string with model name
# path: string with path to save
def save_model( model, model_name, path ):
	# serialize model to JSON
    model_json = model.to_json()
    with open(path + model_name+"_DUO.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()

    # serialize weights to HDF5
    model.save_weights( path + model_name+"_DUO.h5")
    print("Saving files:")
    print(path + model_name+"_DUO.json")
    print(path + model_name+"_DUO.h5")
    print("Model saved to disk") 

# SAVE_MODEL_HISTORY -------------------------------------------------------------------
# saves a trained model into a csv file
# model_hist: history of the model to be saved
# model_name: string with model name
# path: string with path to save
def save_model_history(  model_hist, model_name, path ):
    file = open(path + model_name + "_hist.csv", "w")
    w = csv.writer( file )
  
    for key, val in model_hist.history.items():
        w.writerow([key, val])
    
    file.close()
    print(path + model_name+"_DUO.h5")
    print("Model history saved to disk") 

# LOAD_MODEL_HISTORY ------------------------------------------
# loads a saved model history into memory
# model_name: the name of the model
# path: path to model history
def load_model_history( model_name, path):

    model_hist_loaded = {}
    values = []

    # load dictionary
    r = open( path + model_name + "_hist.csv", "r").read()
    for line in r.split("\n"):
        if(len(line) == 0):
            continue
  
        metric = line.split(",\"[")[0]                                    # extract metrics
        values_str = line.split(",\"[")[1].replace("]\"","").split(", ")  # extract validation values
        values = [float(val_str) for val_str in values_str]
        model_hist_loaded.update( {metric : values} )
    
    return model_hist_loaded

# LOAD_MODEL ------------------------------------------
# loads a saved model into memory
# model_name: the name of the model
# path: path to model history 
def load_model( model_name, path ):
    json_file = open( path + model_name +  "_DUO.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path + model_name +  "_DUO.h5")
    print("Loaded model from disk")
    
    return loaded_model
    
    
# ENCODE_DATA --------------------------------------------------------------------------
# Applies one hot encoder to data
# data: a dataframe
# class_var: string with class variable name
def encode_data(data, class_var):

	feature_names = data.drop([class_var], axis=1).columns.tolist()
	
	X = data[ feature_names ].values
	y = data[class_var].values

	n_features = X.shape[1]
	n_classes = len(data[class_var].unique())
	
	# create numerical encoding for attribute species
	enc = OneHotEncoder()
	Y = enc.fit_transform(y[:, np.newaxis]).toarray()

	# Scale data to have mean 0 and variance 1 
	# which is importance for convergence of the neural network
	scaler = MinMaxScaler()
	X_scaled = scaler.fit_transform(X)
	
	return X_scaled, Y, enc, scaler

# LOAD_TRAINING_DATA ---------------------------------------------------------------------
# loads into a multiarray format a training set previously saved in a .csv file
# dataset_path: string containing the path where the files will be saved
def load_training_data( dataset_path ):
	X_train = pd.read_csv(dataset_path.replace(".csv", "") + "_Xtrain.csv", index_col=False).values
	X_test = pd.read_csv(dataset_path.replace(".csv", "") + "_Xtest.csv", index_col=False).values
	X_validation =pd.read_csv(dataset_path.replace(".csv", "") + "_Xvalidation.csv",index_col=False).values
	Y_train = pd.read_csv(dataset_path.replace(".csv", "") + "_Ytrain.csv",index_col=False).values
	Y_test =pd.read_csv(dataset_path.replace(".csv", "") + "_Ytest.csv", index_col=False).values
	Y_validation = pd.read_csv(dataset_path.replace(".csv", "") + "_Yvalidation.csv", index_col=False).values
	
	return X_train, Y_train, X_test, Y_test, X_validation, Y_validation

# GENERATE_SAVE_TRAINING_DATA ------------------------------------------------------------
# generates training, test and validation sets and stores this information into files
# dataset_path: string containing the path where the files will be saved
# X: NxM matrix representing the training data
# Y: NxC matrix representing the OneHotEconder of C classes
def generate_save_training_data( dataset_path, X, Y):

	# generate train, test and validation sets
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=515)
	X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=515)
	
	np.savetxt(dataset_path.replace(".csv", "") + "_Xtrain.csv", X_train, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Xtest.csv", X_test, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Xvalidation.csv", X_validation, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Ytrain.csv", Y_train, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Ytest.csv", Y_test, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Yvalidation.csv", Y_validation, delimiter=",")

    
