import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import argparse
import os
import sys
import joblib

def main():

    parser = argparse.ArgumentParser(description='Train randomforest.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('save_dir_path', type=str, help='Path to directory to save')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_randomforest.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    df_train = pd.read_csv(args.train_path)
    df_dev_in = pd.read_csv(args.dev_in_path)

    # Extract features and targets
    X_train = df_train.iloc[:,6:]
    X_dev_in = df_dev_in.iloc[:,6:]
    y_train = df_train['fact_cwsm_class']
    y_dev_in = df_dev_in['fact_cwsm_class']


    clf = RandomForestClassifier(n_estimators=50, random_state=args.seed, verbose=3, n_jobs=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_dev_in)

    print("Accuracy:",metrics.accuracy_score(y_dev_in, y_pred))
    
    # Use predict_proba to get probabilities

    joblib.dump(clf, f'{args.save_dir_path}/seed{args.seed}.joblib')


if __name__ == '__main__':
    main()
