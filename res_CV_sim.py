
#%% Load packages 

import pandas as pd
from sklearn.model_selection import train_test_split
from main import *
from cv_hyper_search import *

#%% Data 

# Data train
df = pd.read_csv('data/Sim_data_train.csv')
data_to_split = df.to_numpy()

# Train-Validation split
data_train, data_val = train_test_split(data_to_split, test_size=0.2, shuffle=True)

#%% Optimal hyperparameters for inference models

hyperpar_list = {
    'layer_size_bae': [14, 10, 7],
    'rep_size_bae': [10, 7, 4],
    'drop_bae': [0, 0.1, 0.2],
    'lr_bae': [0.001, 0.0005, 0.0001],
    'n_epochs_bae': [100, 200, 300],
    'b_size_bae': [22, 11, 6],
    'alpha_bae': [0.05, 0.1, 0.5, 1, 5],
    'alpha_sci': [1],                
    'm_sci': [3, 5, 7], 
    'layer_size_sci': [53, 27, 14],
    'lr_sci': [0.001, 0.0005, 0.0001],
    'n_epochs_sci': [100, 200, 300],
    'b_size_sci': [22, 11, 6],
    'layer_size_scinn': [53, 27, 14],
    'lr_scinn': [0.001, 0.0005, 0.0001],
    'n_epochs_scinn': [100, 200, 300],
    'alpha_scw': [0.05, 0.1, 0.5, 1, 5],
    'order_scw': [1],
    'm_scw': [3, 5, 7],
    'alpha_lm': [0, 0.05, 0.1, 0.5, 1, 5],
    'order_lm': [1, 2],
    'n_trees_rf': [100, 200, 300],
    'max_depth_rf': [None],
    'n_features_rf': ['auto', 'sqrt'],
    'min_samples_leaf_rf': [1, 3, 5],
    'layer_size_nn': [53, 27, 14],
    'lr_nn': [0.001, 0.0005, 0.0001],
    'drop_nn': [0, 0.1, 0.2],
    'n_epochs_nn': [100, 200, 300],
    'b_size_nn': [22, 11, 6],
    'n_trees_cf': [100, 200, 300],
    'max_depth_cf': [None],
    'n_features_cf': ['auto', 'sqrt'],
    'min_samples_leaf_cf': [1, 3, 5],
    'layer_size_dr': [53, 27, 14],
    'rep_size_dr': [22, 11, 6],
    'lr_dr': [0.001, 0.0005, 0.0001],
    'E': [5],                          
    'drop_dr': [0, 0.1, 0.2],
    'n_epochs_dr': [100, 200, 300],
    'b_size_dr': [22, 11, 6]   
    }    

models = ['lm', 'rf', 'nn', 'gps', 'dr', 'sci', 'cgct_gps']

hyper_opt_list = []
for i, model in enumerate(models):
    print('i=',i)
    cv_results = CV_hyperpar_search(data_train, data_val, model, hyperpar_list)
    hyperpars_opt = GetOptHyperpar(cv_results)
    hyper_opt_list.append(hyperpars_opt)

