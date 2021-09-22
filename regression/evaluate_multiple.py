'''
Peforms general classification evaluation from a predictions file
(single seed and ensemble results)

Expects to be passed a directory with following contents:

- targets.npy
- mean1.npy
- mean2.npy
 .
 .
 .
- mean10.npy

where each seed prediction file contains a numpy array: [num_samples]

'''

import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import argparse
import os
import sys
from calibrate import gaussian_negative_log_likelihood as gnll

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval regression')
    parser.add_argument('preds_dir1', type=str, help='Path to dir of predictions')
    parser.add_argument('preds_dir2', type=str, help='Path to dir of predictions') 

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate_regression.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # labels
    labels = np.load(f'{args.preds_dir1}/targets.npy')

    # Load preds
    all_means = []
    all_variances = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_means.append(np.load(f'{args.preds_dir1}/mean{seed}.npy'))
        all_variances.append(np.load(f'{args.preds_dir1}/variance{seed}.npy'))    
        all_means.append(np.load(f'{args.preds_dir2}/mean{seed}.npy'))
        all_variances.append(np.load(f'{args.preds_dir2}/variance{seed}.npy'))


    # Ensemble results
    ens_preds = np.mean(np.stack(all_means), axis=0)
    ens_variances = np.mean(np.stack(all_variances), axis=0)
    rmse = math.sqrt(mean_squared_error(labels, ens_preds))
    mae = mean_absolute_error(labels, ens_preds)
    print('---------------')
    print('Ens')
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('---------------')
    nll = gnll(ens_preds, ens_variances, labels)
    print('nll:', nll)
    print('---------------')
    print('---------------')

    # Single seed results
    rmses = []
    maes = []
    nlls = []
    for preds, variances in zip(all_means, all_variances):
        rmses.append(math.sqrt(mean_squared_error(labels, preds)))
        maes.append(mean_absolute_error(labels, preds))
        nlls.append(gnll(preds, variances, labels))
    rmses = np.asarray(rmses)
    maes = np.asarray(maes)
    nlls = np.asarray(nlls)
    print('---------------')
    print('Single')
    print(f'RMSE: {rmses.mean()} +- {rmses.std()}')
    print(f'MAE: {maes.mean()} +- {maes.std()}')
    print('---------------')
    print(f'nll: {nlls.mean()} +- {nlls.std()}')
    print('---------------')
