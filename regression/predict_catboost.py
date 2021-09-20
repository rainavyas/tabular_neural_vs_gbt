'''
Generate prediction files for catboost models in the following format:

Argument specifies destination directory in which this script
generates content with following contents:

- targets.npy
- mean1.npy
- mean2.npy
 .
 .
 .
- mean10.npy
- variance1.npy
.
.
.
- variance2.npy

where each seed prediction file contains a numpy array: [num_samples]
'''

import numpy as np
import catboost
import pandas as pd
import argparse
import os
import sys

def get_predictions(X, model):
    '''
    Calculates predictions on df features for specified model
    
    Return: array [num_samples x 2],
        where
            num_samples = number of rows in features_df
            2 = [mean, standard_deviation]
    
    '''
    return model.predict(X)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='catboost predict.')
    parser.add_argument('trained_models_dir', type=str, help='Path to dir of trained models')
    parser.add_argument('out_dir', type=str, help='Path to dir to save prediction files')
    parser.add_argument('data_paths', type=str, help='list of all eval data files')

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict_catboost.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    data_paths = args.data_paths.split()
    dfs = []
    for data_path in data_paths:
        dfs.append(pd.read_csv(data_path))
    df = pd.concat(dfs)

    print("Loaded Data")

    # 10 models assumed
    ensemble_size = 10

    # Generate prediction file for each model
    for seed in range(1, ensemble_size+1):

        model = catboost.CatBoostRegressor()
        model.load_model(f'{args.trained_models_dir}/seed{seed}.cbm')

        # get targets for dev
        y_dev = df['fact_temperature']
        labels = np.asarray(y_dev)
        np.save(f'{args.out_dir}/targets.npy', labels)

        # get mean and variance predictions
        X = df.iloc[:,6:]
        preds = np.asarray(get_predictions(X, model))
        means = np.squeeze(preds[:,0])
        np.save(f'{args.out_dir}/mean{seed}.npy', means)
        variances = np.squeeze(preds[:,1])**2
        np.save(f'{args.out_dir}/variance{seed}.npy', variances)

        print('Done seed', seed)