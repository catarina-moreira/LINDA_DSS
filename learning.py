
# update for tensorflow
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import seaborn as sns
import random as rn
import re
import warnings
import csv

import tensorflow as tf
# Force TensorFlow to single thread
# Multiple threads are a potential source of non-reprocible research resulsts
session_conf = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1 )

# tf.set_random_seed() will make random number generation in the TensorFlow backend
# have a well defined initial state
# more details: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.compat.v1.set_random_seed(515)

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

# Bayesian networks
from sklearn.preprocessing import KBinsDiscretizer
from pylab import *
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

# for classification purposes
from pyAgrum.lib.bn2roc import showROC

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import pylab as pl
from pylab import savefig
plt.style.use('seaborn-deep')


# RECALL -----------------------------------------------------------------------------
#
def recall_m(y_true, y_pred):
    """Computes the recal measure of an evaluation setting

    Parameters
    ----------
    y_true : list
       list of groundtruth labels
    y_pred : list
        list of predictions from blackbox

    Returns
    -------
    recall : vector
        a vector with the recall values between the predictions and the groundtruths
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# PRECISION ---------------------------------------------------------------------------
#
def precision_m(y_true, y_pred):
    """Computes the precision measure of an evaluation setting

    Parameters
    ----------
    y_true : list
       list of groundtruth labels
    y_pred : list
        list of predictions from blackbox

    Returns
    -------
    precision : vector
        a vector with the precision values between the predictions and the groundtruths
    """
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

def plot_model_history( model_history, metric ):

    plt.plot(model_history[ metric.lower() ], label='train')
    plt.plot(model_history["val_" + metric.lower()], label='validation')
    plt.ylabel(metric)
    plt.xlabel('Number of Epochs')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

def plot_ROC_Curve( model, X, Y, n_classes):

    Y_pred_proba = model.predict(X)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], Y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

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
# 
# dataset_path: string containing the path where the files will be saved
# X: NxM matrix representing the training data
# Y: NxC matrix representing the OneHotEconder of C classes
def generate_save_training_data( dataset_path, X, Y):
	"""Generates training, test and validation sets and stores this information into files 
	
	Parameters
	----------
	dataset_path : str
		The file location of the spreadsheet
	samples : int, optional
		The number of permutations to generate from the original vector (default is 300)
	variance : int, optional
		Quantity to permute in each feature (default is 0.25)
		
	Returns
	-------
	permutations : matrix
		a 2-D matrix with dimensions (samples, features) with all the permutations of the 
		original vector
	"""
	# generate train, test and validation sets
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=515)
	X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=515)
	
	np.savetxt(dataset_path.replace(".csv", "") + "_Xtrain.csv", X_train, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Xtest.csv", X_test, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Xvalidation.csv", X_validation, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Ytrain.csv", Y_train, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Ytest.csv", Y_test, delimiter=",")
	np.savetxt(dataset_path.replace(".csv", "") + "_Yvalidation.csv", Y_validation, delimiter=",")
	
    

##############################################################################
#						BAYESIAN NETWORK EXPLANATIONS						 #
##############################################################################


def compute_perm_range(feat, variance = 0.25):
    """
    
    Parameters
    ----------
    feat : float
    	Value of a feature to be permuted
    samples : int, optional
        The number of permutations to generate from the original vector (default is 300)
    variance : int, optional
    	Quantity to permute in each feature (default is 0.25)
    	
    Returns
    -------
    min_range : float
    	minimum value that a feature can be permuted
    max_range : float
    	maximum value that a feature can be permuted
    """

    min_range = feat - variance
    max_range = feat + variance
    
    # features are scaled between 0 and 1
    # if the permutation make the feature negative, this values is set to 0
    if( min_range < 0 ):
        min_range = 0
    # if the permutation make the feature bigger than 1, this values is set to 1   
    if( max_range > 1 ):
        max_range = 1
        
    return min_range, max_range 


# PERMUTE_SINGLE_FEATURES_____________________________________________________________
# 
def permute_single_features( my_array, samples = 300, variance = 0.25 ):
    """Given a single array from which one pretends to generate local explanations from
    Draw samples from a uniform distribution within a range of feature_val +- variance
    Returns a matrix with a number of samples (by default 300) with permutations 
    of each feature of the input vector
    
    Parameters
    ----------
    my_array : np.array
    	The datapoint to be locally explained
    samples : int, optional
        The number of permutations to generate from the original vector (default is 300)
    variance : int, optional
    	Quantity to permute in each feature (default is 0.25)
    	
    Returns
    -------
    permutations : matrix
    	a 2-D matrix with dimensions (samples, features) with all the permutations of the 
    	original vector
    """

    # permutation result list
    permutations = []
    # just keeping a controlled number of decimal places
    my_array = np.round(my_array,4)
    
    # keep a copy of the original array, since we will be changing the features
    my_array_backup = my_array
    
    # extract number of features
    num_features = my_array.shape[0]
    
    # add original vector to dataframe
    permutations.append( my_array_backup.tolist() )
    
    # for each feature of the input feature vector,
    for feat in range(0, num_features):    
    
    	# get feature value
    	my_array = my_array_backup
    	feature_val = my_array[feat]
    	
    	# set permutation of feature between [ feat - variance ; feat + variance ]
    	min_range, max_range = compute_perm_range( feature_val, variance )
    	
    	# generate sample of random features within a range
    	for perm in range(0, int(round(samples/num_features, 0))):
    		# set the new vector
    		my_array[feat] = np.abs(np.round(rn.uniform(min_range, max_range),4))
    		permutations.append( my_array.tolist() )
    		
    #rn.shuffle(permutations)
    return permutations


def check_input( value ):
  if value < 0:
    return 0
  if value > 1:
    return 1

  return value

def permute_single_features_circle( my_array, samples = 300, variance = 0.25 ):

    # permutation result list
    permutations = []
    # just keeping a controlled number of decimal places
    my_array = np.round(my_array,4)

    # keep a copy of the original array, since we will be changing the features
    my_array_backup = my_array

    # extract number of features
    num_features = my_array.shape[0]

    # add original vector to dataframe
    permutations.append( my_array_backup.tolist() )

    # for each feature of the input feature vector,
    for perm in range(0, int(round(samples/num_features, 0))):
  
      # generate sample of random features within a range
      
      temp1 = []
      temp2 = []
      for feat in range(0, num_features):   
        theta = 2*math.pi*np.random.random()
        feature_val = my_array[feat]

        # set the new vector
        temp1.append( check_input( feature_val + np.round(np.random.uniform(0, variance),4)*math.cos(theta) ))
        temp2.append( check_input( feature_val + np.round(np.random.uniform(0, variance),4)*math.sin(theta) ))

      permutations.append( temp1 )
      permutations.append( temp2 )
        
    #rn.shuffle(permutations)
    return permutations[0:samples]



# LEARNBN -------------------------------------------
#
def learnBN( file_path, algorithm = "Hill Climbing" ):
    """Given a single array from which one pretends to generate local explanations from
    Draw samples from a uniform distribution within a range of feature_val +- variance
    Returns a matrix with a number of samples (by default 300) with permutations 
    of each feature of the input vector
    
    Parameters
    ----------
    my_array : np.array
    	The datapoint to be locally explained
    samples : int, optional
        The number of permutations to generate from the original vector (default is 300)
    variance : int, optional
    	Quantity to permute in each feature (default is 0.25)
    	
    Returns
    -------
    permutations : matrix
    	a 2-D matrix with dimensions (samples, features) with all the permutations of the 
    	original vector
    """

    learner = gum.BNLearner( file_path )
    
    if( algorithm == "Hill Climbing"):
        print("Selecting Greedy Hill Climbing Algorithm")
        learner.useGreedyHillClimbing()
    
    if( algorithm == "Local Search" ):
        print("Selecting Local Search Algorithm")
        bn = learner.useLocalSearchWithTabuList()
        
    if( algorithm == "3off2"):
        print("Selecting 3Off2 Algorithm")
        learner.use3off2()
        
    if( algorithm == "miic" ):
        print("Selecting MIIC Algorithm")
        learner.useMIIC()
        
    learner.learnBN()
    
    bn = learner.learnBN()
    essencGraph = gum.EssentialGraph( bn )
    infoBN = gnb.getInformation( bn )  
    
    return [ bn, infoBN, essencGraph ]

# DISCRETIZE_DATAFRAME -------------------------------------------------------
#
#
def discretize_dataframe( df, class_var, num_bins=4 ):
    """Given a dataframe with continuous values, convert the continuous values into discrete ones
       by splitting the data into bins and by computing the respective quartiles
    
    Parameters
    ----------
    df : pd.DataFrame
    	The datapoint to be locally explained
    class_var : str
        The number of permutations to generate from the original vector (default is 300)
    num_bins : int, optional
    	Quantity to permute in each feature (default is 0.25)
    	
    Returns
    -------
    permutations : matrix
    	a 2-D matrix with dimensions (samples, features) with all the permutations of the 
    	original vector
    """    
    r=np.array(range(num_bins+1))/(1.0*num_bins)
    
    # quantiles are building using pandas.qcut
    # The "class" column is just copied.
    l=[]
    for col in df.columns.values:
        
        if col!=class_var:
            l.append( pd.DataFrame( pd.qcut( df[col],r, duplicates='drop',precision=2),columns=[col]))
        else:
            l.append( pd.DataFrame( df[col].values,columns=[col]))
    
    treated = pd.concat(l, join='outer', axis=1)
    return treated

# SAVE_DISCRETIZED_DATAFRAME ---------------------------------------------------
#
def save_discretized_dataframe(indx, df_model, model_type, perm_type, bins, dataset_name, path, class_var):
    """Given a single array from which one pretends to generate local explanations from
    Draw samples from a uniform distribution within a range of feature_val +- variance
    Returns a matrix with a number of samples (by default 300) with permutations 
    of each feature of the input vector
    
    Parameters
    ----------
    my_array : np.array
    	The datapoint to be locally explained
    samples : int, optional
        The number of permutations to generate from the original vector (default is 300)
    variance : int, optional
    	Quantity to permute in each feature (default is 0.25)
    	
    Returns
    -------
    permutations : matrix
    	a 2-D matrix with dimensions (samples, features) with all the permutations of the 
    	original vector
    """    
    file_path = path + dataset_name + "/" + str(indx) + "/" + re.sub( r"\.\w+", "", dataset_name ) + "_" + model_type +"_INDX_" + str(indx) + "_" + perm_type +".csv"
    df_discr = discretize_dataframe( df_model, bins, class_var )
    
    print("Saving discretized dataset into: %s\n" %(file_path))
    df_discr.to_csv( file_path, index=False)


# WRAP_INFORMATION -------------------------------------------
#
def wrap_information( local_data_dict ):
    
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    for instance in local_data_dict:
        
        # wrap up true positives
        if( instance['prediction_type'] == 'TRUE POSITIVE'):
            true_positives.append(instance)

        # wrap up true negatives
        if( instance['prediction_type'] == 'TRUE NEGATIVE' ):
            true_negatives.append(instance)
        
        # wrap up false positives
        if( instance['prediction_type'] == 'FALSE POSITIVE' ):
            false_positives.append(instance)
        
        # wrap up false negatives
        if( instance['prediction_type'] == 'FALSE NEGATIVE' ):
            false_negatives.append(instance)
            
    return true_positives, true_negatives, false_positives, false_negatives
    

# GENERATE_PERMUTATIONS -------------------------------------------
#
def generate_permutations( instance, labels_lst, feature_names, class_var, encoder, scaler, model, samples = 300, variance = 0.25):
    
    # get datapoint in scaled feature space
    local_datapoint = np.array(instance['scaled_vector'])
    # get datapoint in original feature space
    local_datapoint_orig = np.array(instance['original_vector'])
    
    # permute features
    permutations = permute_single_features( local_datapoint, samples = samples, variance = variance )
    #permutations = permute_single_features_circle( local_datapoint, samples = samples, variance = variance )
    
    # convert permutations to original feature space
    permutations_orig = scaler.inverse_transform( permutations )
    
    # compute predictions for each permuted instance
    predictions = encoder.inverse_transform( model.predict( permutations ) )
    
    # convert prediction classes to labels
    labelled_predictions = [ labels_lst[ int(predictions[indx][0]) ] for indx in range(0, len(predictions))]
    
    # add all this information to a single dataframe
    df_local_permutations = pd.DataFrame( permutations_orig, columns = feature_names )

    # add class variable to dataframe
    df_local_permutations[ class_var ] = labelled_predictions
    
    return df_local_permutations

# GEBERATE_BN_EXPLANATIONS ------------------------------------------------------------
#
def generate_BN_explanations(instance, label_lst, feature_names, class_var, encoder, scaler, model, path, dataset_name ):

    # necessary for starting Numpy generated random numbers in an initial state
    np.random.seed(515)

    # Necessary for starting core Python generated random numbers in a state
    rn.seed(515)

    indx = instance['index']
    prediction_type = instance['prediction_type'].lower()+"s"
    prediction_type = prediction_type.replace(" ", "_")
    
    # generate permutations
    df = generate_permutations( instance, label_lst, feature_names, class_var, encoder, scaler, model)

    # discretize data
    df_discr = discretize_dataframe( df, class_var, num_bins=4 )

    # save discretised dataframe (for debugging and reproduceability purposes)
    path_to_permutations = path + "feature_permutations/" + dataset_name.replace(".csv","") + "/" + prediction_type  + "/" + str(indx) + ".csv"
    df_discr.to_csv( path_to_permutations, index=False)

    # normalise dataframe
    normalise_dataframe( path_to_permutations )

    # learn BN
    bn, infoBN, essencGraph = learnBN( path_to_permutations.replace(".csv", "_norm.csv") )

    # perform inference
    inference = gnb.getInference(bn, evs={},targets=df_discr.columns.to_list(), size='12')

    # show networks
    gnb.sideBySide(*[bn, inference, infoBN  ],
             captions=[ "Bayesian Network", "Inference", "Information Network" ])

    # save to file
    path_to_explanation = path + "explanations/" + dataset_name.replace(".csv", "") + "/BN/" + prediction_type + "/"
    gum.lib.bn2graph.dotize( bn , path_to_explanation + str(indx) + "_BN" )
    gum.saveBN(bn,path_to_explanation + str(indx) + "_BN.net" )

    return [bn, inference, infoBN]
 
 
# GEBERATE_BN_EXPLANATIONSMB ------------------------------------------------------------
#
def generate_BN_explanationsMB(instance, label_lst, feature_names, class_var, encoder, scaler, model, path, dataset_name, variance = 0.1, algorithm = "Hill Climbing"  ):

    # necessary for starting Numpy generated random numbers in an initial state
    np.random.seed(515) 

    # Necessary for starting core Python generated random numbers in a state
    rn.seed(515)

    indx = instance['index']
    prediction_type = instance['prediction_type'].lower()+"s"
    prediction_type = prediction_type.replace(" ", "_")
    
    # generate permutations
    df = generate_permutations( instance, label_lst, feature_names, class_var, encoder, scaler, model, variance = variance)

    # discretize data
    df_discr = discretize_dataframe( df, class_var, num_bins=4 )

    # save discretised dataframe (for debugging and reproduceability purposes)
    path_to_permutations = path + "feature_permutations/" + dataset_name.replace(".csv","") + "/" + prediction_type  + "/" + str(indx) + ".csv"
    df_discr.to_csv( path_to_permutations, index=False)

    # normalise dataframe
    normalise_dataframe( path_to_permutations )

    # learn BN
    bn, infoBN, essencGraph = learnBN( path_to_permutations.replace(".csv", "_norm.csv"), algorithm = algorithm)

    # perform inference
    inference = gnb.getInference(bn, evs={},targets=df_discr.columns.to_list(), size='12')
    
    # compute Markov Blanket
    markov_blanket = gum.MarkovBlanket(bn, class_var)
    
    # show networks
    # gnb.sideBySide(*[bn, inference, markov_blanket  ],
    #         captions=[ "Bayesian Network", "Inference", "Markov Blanket" ])

    # save to file
    path_to_explanation = path + "explanations/" + dataset_name.replace(".csv", "") + "/BN/" + prediction_type + "/"
    gum.lib.bn2graph.dotize( bn , path_to_explanation + str(indx) + "_BN" )
    gum.saveBN(bn,path_to_explanation + str(indx) + "_BN.net" )

    return [bn, inference, infoBN, markov_blanket]
 

# GENERATE_LOCAL_PREDICTIONS -------------------------------------------
#
def generate_local_predictions( X, Y, model, scaler, encoder ):
    
    # get original vector
    orig_vec = np.round(scaler.inverse_transform(X),6)

    # generate all predictions for X
    predictions = model.predict( X )

    # extrace the label of the prediction of X[indx]
    prediction_class = encoder.inverse_transform( predictions )
    local_data_dict = []
    for indx in range(0, orig_vec.shape[0]):

        ground_truth = np.expand_dims(Y[indx], axis=0)
        ground_truth_class = encoder.inverse_transform( ground_truth )[0][0]

        prediction = prediction_class[indx][0]

        # check if data point is a true positive
        if( ( int(prediction) == int(ground_truth_class) ) & (int(prediction)==1) & (int(ground_truth_class)==1) ):
            pred_type = "TRUE POSITIVE"

        # check if data point is a true negative
        if( ( int(prediction) == int(ground_truth_class) ) & (int(prediction)==0) & (int(ground_truth_class)==0) ):
            pred_type = "TRUE NEGATIVE"

        # check if data point is a false negative
        if( ( int(prediction) != int(ground_truth_class) ) & (int(prediction)==0) & (int(ground_truth_class)==1) ):
            pred_type = "FALSE NEGATIVE"

        # check if data point is a false positve
        if( ( int(prediction) != int(ground_truth_class) ) & (int(prediction)==1) & (int(ground_truth_class)==0) ):
            pred_type = "FALSE POSITIVE"

        local_data_dict.append( {'index' : indx,
                                 'original_vector' : orig_vec[indx,:].tolist(),
                                 'scaled_vector' : X[indx,:].tolist(),
                                 'ground_truth' : ground_truth_class,
                                 'predictions' : prediction,
                                 'prediction_type' : pred_type})
    return local_data_dict
    
##################################################################################
# 					TEXT PROCESSING												 #
# ###############################################################################


# FIND -----------------------------------------------
# 
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

# UNTOKENIZE -----------------------------------------------
#
def untokenize( tokens, delim ):
    
    untokenized = tokens[0]
    
    for indx in range(1, len(tokens)):
        untokenized = untokenized + delim + tokens[indx]
        
    return untokenized

# NORMALISE_LINE -------------------------------------------
#
def normalise_line( my_str, class_label  ):
    
    my_str = my_str.replace("\","+class_label, "")
    my_str = my_str.replace("-1e-05", "0.0000")
    
    tokens = my_str.split("\",\"")
    tokens_norm = []

    for token in tokens:

        token = token.replace("]","")

        indxs = find(token, ".")
        indx_comma = find(token, ",")[0]+2

        if( (len(token[indxs[1]+1 : -1 ]) >= 4) & (len( token[indxs[0]+1 : indx_comma-2 ]) >= 4)  ):
            token_temp = token[0:indxs[0]] + "." + token[indxs[0] + 1 : indxs[0]+5] + ", " +token[indx_comma:indxs[1]] + token[indxs[1] : indxs[1]+5  ] + "]"
        
        if( (len(token[indxs[1]+1 : -1 ]) < 4) & (len( token[indxs[0]+1 : indx_comma-2 ]) >= 4) ):
            extra =  "0"*(np.abs(len(token[indxs[1]+1 : -1 ]) - 4))
            token_temp = token[0:indxs[0]] + "." + token[indxs[0] + 1 : indxs[0]+5] + ", " +token[indx_comma:indxs[1]]  + token[indxs[1] : -1 ] + extra + "]"
         
        if( (len(token[indxs[1]+1 : -1 ]) >= 4) & (len( token[indxs[0]+1 : indx_comma-2 ]) < 4) ):
            extra =  "0"*(np.abs(len( token[indxs[0]+1 : indx_comma-2 ]) - 4))
            token_temp = token[0:indxs[0]] + "." + extra + ", " +token[indx_comma:indxs[1]]  + token[indxs[1] : -1 ] + extra + "]"
        
        if( (len(token[indxs[1]+1 : -1 ]) < 4) & (len( token[indxs[0]+1 : indx_comma-2 ]) < 4) ):
            extra2 = "0"*(np.abs(len(token[indxs[1]+1 : -1 ]) - 4))
            extra1 = "0"*(np.abs(len(token[indxs[0]+1 : -1 ]) - 4))
            token_temp = token[0:indxs[0]] + "." + extra1 + ", " +token[indx_comma:indxs[1]]  + token[indxs[1] : -1 ] + extra2 + "]"
    
        tokens_norm.append(token_temp)

    return untokenize( tokens_norm, "\",\"") + "\"," +class_label

# NORMALISE_LINE -------------------------------------------
#
def normalise_dataframe( path_to_permutations ):
    file = open(path_to_permutations,"r")

    f_write = open(path_to_permutations.replace(".csv", "_norm.csv"),"w")

    header = file.readline().replace("\n","")
    f_write.write( header + "\n")
    
    for line in file.readlines():
        
        # get class
        class_label = line.split("\",")[-1].replace("\n","")
        # normalise dataframe input
        line_norm = normalise_line( line.replace("\n",""), class_label  )
        # write normalised input to file
        f_write.write(line_norm + "\n")
    
    file.close()
    f_write.close()
    


 


