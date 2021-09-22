import argparse
import os
import sys
import pandas as pd
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
import pickle

def main():

    parser = argparse.ArgumentParser(description='Train ngboost.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('save_dir_path', type=str, help='Path to directory to save')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
    parser.add_argument('--depth', type=int, default=3, help='Specify the max depth')
    parser.add_argument('--lr', type=float, default=3, help='Specify the learning rate')

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

    model_ngb = NGBRegressor(n_estimators=650,  natural_gradient=True, learning_rate = 0.01,Base=default_tree_learner, Dist=Normal, Score=MLE, verbose_eval =50)
    model_ngb.fit(X_train, y_train, X_val = X_dev_in, Y_val = y_dev_in, early_stopping_rounds = 10)

    #with open(f'{args.save_dir_path}/seed{args.seed}.p', 'wb') as f:
    #    pickle.dump(ngb, f)

if __name__ == '__main__':
    main()
