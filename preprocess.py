'''
Preprocess the tabular data to:

1) normalize using the training data
2) Replace NaNs with training data means for continuous features and modes for categorical features

Expects a directory at the input with the following files:

- dev_in.csv
- dev_out.csv
- eval_in.csv
- eval_out.csv

'''

import pandas as pd
import numpy as np
import argparse
import os
import sys
from scipy import stats
from typing import Dict
import sklearn.preprocessing


def normalize(
    X: Dict[str, np.ndarray], normalization: str, seed: int, noise: float = 1e-3
) -> Dict[str, np.ndarray]:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        if noise:
            X_train = X_train.copy()
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)
            X_train += noise_std * np.random.default_rng(seed).standard_normal(
                X_train.shape
            )
    else:
        raise ValueError(f'unknown normalization: {normalization}')
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='catboost predict.')
    parser.add_argument('in_dir', type=str, help='Path to dir of original data')
    parser.add_argument('out_dir', type=str, help='Path to dir to save processed data')
    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/preprocess.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load in data
    df_train = pd.read_csv(f'{args.in_dir}/train.csv')
    df_dev_in = pd.read_csv(f'{args.in_dir}/dev_in.csv')
    df_dev_out = pd.read_csv(f'{args.in_dir}/dev_out.csv')
    df_eval_in = pd.read_csv(f'{args.in_dir}/eval_in.csv')
    df_eval_out = pd.read_csv(f'{args.in_dir}/eval_out.csv')
    print("Loaded Data")
    print(df_train.head())

    # Identify the categorical features
    cat_features = []
    for col in df_train:
        values = df_train[col].tolist()
        unique = list(dict.fromkeys(values))
        if len(unique) < 20:
            cat_features.append(col)

    nan_replacements = {}
    for col in df_train:
        if col in cat_features:
            nan_replacements[col] = stats.mode(np.asarray(df_train[col].tolist()))[0][0]
        else:
            nan_replacements[col] = np.mean(np.asarray(df_train[col].dropna().tolist()))
    
    # Replaces all NaNs
    for col in df_train:
        df_train[col] = df_train[col].fillna(nan_replacements[col])
        df_dev_in[col] = df_dev_in[col].fillna(nan_replacements[col])
        df_dev_out[col] = df_dev_out[col].fillna(nan_replacements[col])
        df_eval_in[col] = df_eval_in[col].fillna(nan_replacements[col])
        df_eval_out[col] = df_eval_out[col].fillna(nan_replacements[col])
    print("Replaced NaNs")

    # Normalize using training stats
    # Quantile normalisation is used (maps to a normal distribution)
    X_train_np = np.asarray(df_train.iloc[:,6:])
    X_dev_in_np = np.asarray(df_dev_in.iloc[:,6:])
    X_dev_out_np = np.asarray(df_dev_out.iloc[:,6:])
    X_eval_in_np = np.asarray(df_eval_in.iloc[:,6:])
    X_eval_out_np = np.asarray(df_eval_out.iloc[:,6:])

    X = {'train': X_train_np, 'dev_in': X_dev_in_np, 'dev_out': X_dev_out_np, 'eval_in': X_eval_in_np, 'eval_out': X_eval_out_np}
    X = normalize(X, normalization='quantile', seed=100)

    df_train.loc[:,6:] = X['train']
    df_dev_in.loc[:,6:] = X['dev_in']
    df_dev_out.loc[:,6:] = X['dev_out']
    df_eval_in.loc[:,6:] = X['eval_in']
    df_eval_out.loc[:,6:] = X['eval_out']

    print('Normalized')
    print(df_train.head())

    # Save modified dataframes
    df_train.to_csv(f'{args.out_dir}/train.csv', index=False)
    df_dev_in.to_csv(f'{args.out_dir}/dev_in.csv', index=False)
    df_dev_out.to_csv(f'{args.out_dir}/dev_out.csv', index=False)
    df_eval_in.to_csv(f'{args.out_dir}/eval_in.csv', index=False)
    df_eval_out.to_csv(f'{args.out_dir}/eval_out.csv', index=False)
