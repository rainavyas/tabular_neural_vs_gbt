import argparse
import os
import sys
import pandas as pd
import catboost

def main():

    parser = argparse.ArgumentParser(description='Train catboost.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('save_dir_path', type=str, help='Path to directory to save')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_catboost.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    df_train = pd.read_csv(args.train_path)
    df_dev_in = pd.read_csv(args.dev_in_path)

    # Extract features and targets
    X_train = df_train.iloc[:,6:]
    X_dev_in = df_dev_in.iloc[:,6:]
    y_train = df_train['fact_cwsm_class']
    y_dev_in = df_dev_in['fact_cwsm_class']

    # Set training hyperparameters
    depth=6
    iterations=10000
    learning_rate=0.4

    model = catboost.CatBoostClassifier(
        learning_rate = learning_rate,
        iterations = iterations,
        depth = depth,
        loss_function = 'MultiClass',
        eval_metric = 'Accuracy',
        random_seed = args.seed)
    
    model.fit(
        X_train,
        y_train,
        verbose = 20,
        eval_set = (X_dev_in, y_dev_in))
    
    model.save_model(f'{args.save_dir_path}/seed{args.seed}.cbm')

if __name__ == '__main__':
    main()