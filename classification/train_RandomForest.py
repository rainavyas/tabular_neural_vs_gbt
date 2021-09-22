import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import argparse
import os
import sys
import joblib
import numpy as np

def main():

    parser = argparse.ArgumentParser(description='Train randomforest.')
    parser.add_argument('data_dir', type=str, help='Path to train data')
    parser.add_argument('save_dir', type=str, help='Path to directory to save predictions')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_randomforest.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    np.random.seed(args.seed)

    lab_to_ind = {0.0: 0, 10.0: 1, 11.0: 2, 12.0: 3, 13.0: 4, 20.0: 5, 21.0: 6, 22.0: 7, 23.0: 8}

    df_train = pd.read_csv(args.data_dir+'/train.csv')
    df_dev_in = pd.read_csv(args.data_dir+'/dev_in.csv')

    # Extract features and targets
    X_train = df_train.iloc[:,6:]
    X_dev_in = df_dev_in.iloc[:,6:]
    y_train = np.asarray([lab_to_ind[lab] for lab in df_train['fact_cwsm_class']])
    y_dev_in = np.asarray([lab_to_ind[lab] for lab in df_dev_in['fact_cwsm_class']])


    clf = RandomForestClassifier(n_estimators=200, random_state=args.seed, verbose=3, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_dev_in)

    print("Accuracy:",metrics.accuracy_score(y_dev_in, y_pred))
    
    df_dev_out = pd.read_csv(args.data_dir+'/dev_out.csv')
    df_eval_in = pd.read_csv(args.data_dir+'/eval_in.csv')
    df_eval_out = pd.read_csv(args.data_dir+'/eval_out.csv')

    X_dev_out = df_dev_out.iloc[:,6:]
    y_dev_out = np.asarray([lab_to_ind[lab] for lab in df_dev_out['fact_cwsm_class']])
    X_eval_in = df_eval_in.iloc[:,6:]
    y_eval_in = np.asarray([lab_to_ind[lab] for lab in df_eval_in['fact_cwsm_class']])
    X_eval_out = df_eval_out.iloc[:,6:]
    y_eval_out = np.asarray([lab_to_ind[lab] for lab in df_eval_out['fact_cwsm_class']])


    # Use predict_proba to get probabilities

    dev_in_probs = clf.predict_proba(X_dev_in)
    dev_out_probs = clf.predict_proba(X_dev_out)
    eval_in_probs = clf.predict_proba(X_eval_in)
    eval_out_probs = clf.predict_proba(X_eval_out)
    
    np.save(args.save_dir+"/dev_in/"+str(args.seed)+".npy", np.asarray(dev_in_probs))
    np.save(args.save_dir+"/dev_out/"+str(args.seed)+".npy", np.asarray(dev_out_probs))
    np.save(args.save_dir+"/eval_in/"+str(args.seed)+".npy", np.asarray(eval_in_probs))
    np.save(args.save_dir+"/eval_out/"+str(args.seed)+".npy", np.asarray(eval_out_probs))

    np.save(args.save_dir+"/dev_in/targets.npy", np.asarray(y_dev_in))
    np.save(args.save_dir+"/dev_out/targets.npy", np.asarray(y_dev_out))
    np.save(args.save_dir+"/eval_in/targets.npy", np.asarray(y_eval_in))
    np.save(args.save_dir+"/eval_out/targets.npy", np.asarray(y_eval_out))

if __name__ == '__main__':
    main()
