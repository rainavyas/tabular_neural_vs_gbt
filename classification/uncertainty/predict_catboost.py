'''
Generate prediction files for catboost models in the following format:

Argument specifies destination directory in which this script
generates content with following contents:

- targets.npy
- 1.npy
- 2.npy
 .
 .
 .
- 10.npy

where each seed prediction file contains a numpy array: [num_samples x num_classes]

Also generate logit prediction files:

- logits1.npy
- logits2.npy
.
.
.
- logits10.npy
where each file is a numpy array: [num_samples x num_classes]

'''

import argparse
import os
import sys
import numpy as np
import pandas as pd
import catboost

def get_logits(X, model):
    '''
    Returns raw scores predicted by the model
    [num_samples x num_classes]
    '''
    return model.predict(X, prediction_type='RawFormulaVal')

def get_predictions(X, model):
    '''
    Returns predictions for specified model
    Classification: [num_samples x num_classes]
    '''
    return model.predict_proba(X)

def get_lab_to_ind(data_df):
    '''
    Prepare a label to index map
    '''
    y_fact = set(list(data_df['fact_cwsm_class']))
    lab_to_ind = {}
    for i, lab in enumerate(y_fact):
        lab_to_ind[lab] = i
    return lab_to_ind

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
       
        model = catboost.CatBoostClassifier()
        model.load_model(f'{args.trained_models_dir}/seed{seed}.cbm')

        lab_to_ind = get_lab_to_ind(df)

        # get targets for dev
        y_dev = df['fact_cwsm_class']
        labels = np.asarray([lab_to_ind[lab] for lab in y_dev])
        np.save(f'{args.out_dir}/targets.npy', labels)

        # get prediction
        X = df.iloc[:,6:]
        preds = np.asarray(get_predictions(X, model))
        np.save(f'{args.out_dir}/{seed}.npy', preds)

        # get logits
        logits = np.asarray(get_logits(X, model))
        np.save(f'{args.out_dir}/logits{seed}.npy', logits)

        print('Done seed', seed)