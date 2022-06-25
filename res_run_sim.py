
#%% Load packages 

import numpy as np
import pandas as pd
import ast
import torch
from main import *
from cv_hyper_search import *

#%% Data 

# Data train
df = pd.read_csv('data/Sim_data_train.csv')
data_train = df.to_numpy()

# Data test counterfactual
df = pd.read_csv('data/Sim_data_test_f.csv')
data_test_f = df.to_numpy()

# Data test counterfactual
df = pd.read_csv('data/Sim_data_test_cf.csv')
data_test_cf = df.to_numpy()

# Hyperpar list
hyper_opt_list = open("hyper_opt_list_HIV2017_sim.txt", "r")
hyper_opt_list = hyper_opt_list.read()
hyper_opt = ast.literal_eval(hyper_opt_list)

# Convert hyperpar_opt_list so that its values are iterable
for i in range(len(hyper_opt)):
    for key in hyper_opt[i].keys():
        hyper_opt[i][key] = [hyper_opt[i][key]]


#%% Factual point estimate and variance (averaged over 10 runs)

models = ['lm', 'rf', 'nn', 'gps', 'cf', 'dr', 'sci', 'scct_gps', 'scct_dr']

# Set all seeds
np.random.seed(0)
torch.manual_seed(0)


# Get results
res_table = np.empty(shape=(9,10))
for l in range(10):
    test_loss = []
    for i, model in enumerate(models):
        print('i=',i)   
        cv_results = CV_hyperpar_search(data_train, data_test_f, model, hyper_opt[i])
        test_loss.append(cv_results[0]['loss'])
    res_table[:,l] = np.array(test_loss)

#%% Counterfactual point estimate and variance (averaged over 10 runs)

models = ['lm', 'rf', 'nn', 'gps', 'cf', 'dr', 'sci', 'scct_gps', 'scct_dr']

# Set all seeds
np.random.seed(0)
torch.manual_seed(0)


# Get results
res_table = np.empty(shape=(9,10))
for l in range(10):
    test_loss = []
    for i, model in enumerate(models):
        print('i=',i)   
        cv_results = CV_hyperpar_search(data_train, data_test_cf, model, hyper_opt[i])
        test_loss.append(cv_results[0]['loss'])
    res_table[:,l] = np.array(test_loss)
