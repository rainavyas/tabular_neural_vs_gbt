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

and

- variance1.npy
- variance2.npy
 .
 .
 .
- variance10.npy
'''

import numpy as np
import argparse
import os
import sys
import math
from uncertainty import ensemble_uncertainties_regression
from assessment import calc_uncertainty_regection_curve, f_beta_metrics
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval regression')
    parser.add_argument('preds_dir', type=str, help='Path to dir of predictions')

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate_uncertainty.cmd', 'a') as f:
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
    
    # Convert to array: [num_models, num_examples, 2]
    all_means = np.stack(all_means)
    all_variances = np.stack(all_variances)
    all_preds = np.stack((all_means, all_variances), axis=-1)

    # get all uncertainties
    all_ens_uncertainty = ensemble_uncertainties_regression(all_preds)

    # Get errors
    all_preds_mean = all_preds[:,:,0]
    avg_preds = np.squeeze(np.mean(all_preds_mean, axis=0))
    ens_errors = (avg_preds - labels) ** 2

    # Get metric (R-AUC, F1-AUC, F1-95) for each uncertainty measure
    thresh = 1.0
    print('----------------')
    for measure in ['tvar']:
    # for measure in all_ens_uncertainty.keys():
        ens_uncertainties = all_ens_uncertainty[measure]
        rejection_mse = calc_uncertainty_regection_curve(ens_errors, ens_uncertainties)
        ens_r_auc = rejection_mse.mean()
        ens_f_auc, ens_f95, _ = f_beta_metrics(ens_errors, ens_uncertainties, thresh)

        # single_r_aucs = []
        # single_f_aucs = []
        # single_f95s = []
        # for seed in range(10):
        #     print(seed)
        #     single_preds = np.expand_dims(all_preds[seed,:,:], axis=0)
        #     all_single_uncertainty = ensemble_uncertainties_regression(single_preds)
        #     single_uncertainties = all_single_uncertainty[measure]
        #     single_preds_mean = single_preds[:,:,0]
        #     single_avg_preds = np.squeeze(np.mean(single_preds_mean, axis=0))
        #     single_errors = (single_avg_preds - labels) ** 2
        #     rejection_mse = calc_uncertainty_regection_curve(single_errors, single_uncertainties)
        #     single_r_aucs.append(rejection_mse.mean())
        #     single_f_auc, single_f95, _ = f_beta_metrics(single_errors, single_uncertainties, thresh)
        #     single_f_aucs.append(single_f_auc)
        #     single_f95s.append(single_f95)
        # single_r_aucs = np.asarray(single_r_aucs)
        # single_f_aucs = np.asarray(single_f_aucs)
        # single_f95s = np.asarray(single_f95s)
        print()
        print(f'Measure {measure}')
        print(f'Ensemble R-AUC: {ens_r_auc}')
        # print(f'Single R-AUC: {single_r_aucs.mean()} +- {single_r_aucs.std()}')
        print(f'Ensemble F1-AUC: {ens_f_auc}')
        # print(f'Single F1-AUC: {single_f_aucs.mean()} +- {single_f_aucs.std()}')
        print(f'Ensemble F1@95: {ens_f95}')
        # print(f'Single F1@95: {single_f95s.mean()} +- {single_f95s.std()}')
        print()
    print('----------------')
