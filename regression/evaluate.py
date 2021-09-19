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
#from calibration.calibrate import eval_calibration

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval regression')
    parser.add_argument('preds_dir', type=str, help='Path to dir of predictions') 

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate_regression.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # labels
    labels = np.load(f'{args.preds_dir}/targets.npy')

    # Load preds
    all_means = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_means.append(np.load(f'{args.preds_dir}/mean{seed}.npy'))
    
    # Ensemble results
    ens_preds = np.mean(np.stack(all_means), axis=0)
    rmse = math.sqrt(mean_squared_error(labels, ens_preds))
    mae = mean_absolute_error(labels, ens_preds)
    print('---------------')
    print('Ens')
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('---------------')
    #nll, _, ece, mce = eval_calibration(labels, ens_preds)
    #print('nll:', nll)
    #print('---------------')
    #print('---------------')

    # Single seed results
    rmses = []
    maes = []
    #nlls, eces, mces = [], [], []
    for preds in all_means:
        rmses.append(math.sqrt(mean_squared_error(labels, preds)))
        maes.append(mean_absolute_error(labels, preds))
        #nll, _, ece, mce = eval_calibration(labels, preds)
        #nlls.append(nll)
        #eces.append(ece)
        #mces.append(mce)
    rmses = np.asarray(rmses)
    maes = np.asarray(maes)
    #nlls = np.asarray(nlls)
    #eces = np.asarray(eces)
    #mces = np.asarray(mces)
    print('---------------')
    print('Single')
    print(f'RMSE: {rmses.mean()} +- {rmses.std()}')
    print(f'MAE: {maes.mean()} +- {maes.std()}')
    print('---------------')
    #print('nll:', nlls.mean(), nlls.std())
    #print('ece:', eces.mean(), eces.std())
    #print('mce:', mces.mean(), mces.std())
    #print('---------------')
    #print(f'nll: {nlls.mean()} +- {nlls.std()}')
    #print(f'ece: {eces.mean()} +- {eces.std()}')
    #print(f'mce: {mces.mean()} +- {mces.std()}')
    print('---------------')
