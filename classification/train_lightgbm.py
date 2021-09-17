import pandas as pd
import numpy as np
import argparse
import os
import sys
from lightgbm import LGBMClassifier
import pickle

def get_lab_to_ind(data_df):
    '''
    Prepare a label to index map
    '''
    y_fact = set(list(data_df['fact_cwsm_class']))
    lab_to_ind = {}
    for i, lab in enumerate(y_fact):
        lab_to_ind[lab] = i
    return lab_to_ind

def main():

    parser = argparse.ArgumentParser(description='Train lgbm.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('save_dir_path', type=str, help='Path to directory to save')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_lightgbm.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    df_train = pd.read_csv(args.train_path)
    df_dev_in = pd.read_csv(args.dev_in_path)

    lab_to_ind = get_lab_to_ind(df_train)

    # Train
    X_train = np.asarray(df_train.iloc[:,6:])
    y_train = np.asarray(df_train['fact_cwsm_class'])
    y_train = np.asarray([lab_to_ind[lab] for lab in y_train])

    # Dev in
    X_dev_in = np.asarray(df_dev_in.iloc[:,6:])
    y_dev_in = df_dev_in['fact_cwsm_class']
    y_dev_in = np.asarray([lab_to_ind[lab] for lab in y_dev_in])

    # Define the model
    params = {
        'boosting_type':'gbdt',
        'objective':'multiclass',
        'colsample_bytree':0.75,
        'learning_rate':0.4,
        'num_leaves':64,
        'subsample_freq':1,
        'random_state':args.seed,
        'reg_alpha':3,
        'subsample':0.8,
        'verbosity':1,
        'n_estimators':100 # number of epochs
    }

    model = LGBMClassifier(**params)

    # Train
    model.fit(X_train, y_train, eval_metric=['multi_logloss', 'multi_error'], eval_set=[(X_train, y_train), (X_dev_in, y_dev_in)], early_stopping_rounds=10, verbose=1)

    # Save
    model.save_model(f'{args.save_dir_path}/seed{args.seed}.txt', num_iteration=model.best_iteration)

if __name__ == '__main__':
    main()