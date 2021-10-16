import argparse
import os
import sys
import pandas as pd
import catboost

def main():

    parser = argparse.ArgumentParser(description='Train catboost.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('xclim_train_path', type=str, help='Path to xclim_train data')
    parser.add_argument('xtime_train_path', type=str, help='Path to xtime_train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('xclim_dev_path', type=str, help='Path to xclim_dev data')
    parser.add_argument('xtime_dev_path', type=str, help='Path to xtime_dev data')
    parser.add_argument('save_dir_path', type=str, help='Path to directory to save')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_catboost.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    df_train1 = pd.read_csv(args.train_path)
    df_train2 = pd.read_csv(args.xclim_train_path)
    df_train3 = pd.read_csv(args.xtime_train_path)
    df_train = pd.concat([df_train1, df_train2, df_train3])

    df_dev1 = pd.read_csv(args.dev_in_path)
    df_dev2 = pd.read_csv(args.xclim_dev_path)
    df_dev3 = pd.read_csv(args.xtime_dev_path)
    df_dev_in = pd.concat([df_dev1, df_dev2, df_dev3])

    # Extract features and targets
    X_train = df_train.iloc[:,6:]
    X_dev_in = df_dev_in.iloc[:,6:]
    y_train = df_train['fact_temperature']
    y_dev_in = df_dev_in['fact_temperature']

    # Set training hyperparameters
    depth=8
    iterations=20000
    learning_rate=0.4

    model = catboost.CatBoostRegressor(
        learning_rate = learning_rate,
        iterations = iterations,
        depth = depth,
        loss_function = 'RMSEWithUncertainty',
        eval_metric = 'RMSE',
        random_seed = args.seed)
    
    model.fit(
        X_train,
        y_train,
        verbose = 20,
        eval_set = (X_dev_in, y_dev_in))
    
    model.save_model(f'{args.save_dir_path}/seed{args.seed}.cbm')

if __name__ == '__main__':
    main()