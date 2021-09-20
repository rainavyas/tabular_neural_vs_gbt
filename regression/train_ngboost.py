import argparse
import os
import sys
import pandas as pd
from ngboost import NGBRegressor
import pickle

def main():

    parser = argparse.ArgumentParser(description='Train ngboost.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('save_dir_path', type=str, help='Path to directory to save')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_ngboost.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    df_train = pd.read_csv(args.train_path)
    df_dev_in = pd.read_csv(args.dev_in_path)

    # Extract features and targets
    X_train = df_train.iloc[:,6:]
    X_dev_in = df_dev_in.iloc[:,6:]
    y_train = df_train['fact_temperature']
    y_dev_in = df_dev_in['fact_temperature']


    # Define parameters
    params = {
        'Dist' : 'Normal',
        'max_depth' : 4,
        'max_features' : None, 
        'max_leaf_nodes' : None,
        'min_impurity_decrease' : 0.0,
        'min_impurity_split' : None,
        'min_samples_leaf' : 1,
        'min_samples_split' : 2,
        'min_weight_fraction' : 0.0,
        'random_state' : args.seed,
        'minibatch_frac' : 0.5,
        'n_estimators' : 500,
        'learning_rate' : 0.05
    }
    ngb = NGBRegressor(**params)
    ngb.fit(X_train, y_train, X_val=X_dev_in, Y_val=y_dev_in)

    with open(f'{args.save_dir_path}/seed{args.seed}.p', 'wb') as f:
        pickle.dump(ngb, f)

if __name__ == '__main__':
    main()