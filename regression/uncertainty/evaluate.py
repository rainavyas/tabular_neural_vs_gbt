'''
Peforms general regression evaluation from a predictions file
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval regression')
    parser.add_argument('preds_dir', type=str, help='Path to dir of predictions') 

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # labels
    labels = np.load(f'{args.preds_dir}/targets.npy')

    # Load preds
    all_means = []
    all_variances = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_means.append(np.load(f'{args.preds_dir}/mean{seed}.npy'))
        all_variances.append(np.load(f'{args.preds_dir}/variance{seed}.npy'))    

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

    # Single seed results
    rmses = []
    maes = []
    for preds, variances in zip(all_means, all_variances):
        rmses.append(math.sqrt(mean_squared_error(labels, preds)))
        maes.append(mean_absolute_error(labels, preds))
    rmses = np.asarray(rmses)
    maes = np.asarray(maes)
    print('---------------')
    print('Single')
    print(f'RMSE: {rmses.mean()} +- {rmses.std()}')
    print(f'MAE: {maes.mean()} +- {maes.std()}')
    print('---------------')
