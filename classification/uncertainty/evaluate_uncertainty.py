'''
Peforms general classification evaluation from a predictions file
(single seed and ensemble results)

Expects to be passed a directory with following contents:

- targets.npy
- 1.npy
- 2.npy
 .
 .
 .
- 10.npy

where each seed prediction file contains a numpy array: [num_samples x num_classes]
'''

import numpy as np
import argparse
import os
import sys
import math
from uncertainty import ensemble_uncertainties_classification
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
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_dir}/{seed}.npy'))
    all_preds = np.stack(all_preds, axis=0)

    # get all uncertainties
    all_ens_uncertainty = ensemble_uncertainties_classification(all_preds)
    all_ens_uncertainty['confidence'] = all_ens_uncertainty['confidence'] * (-1.0)

    # Get errors
    ens_preds = np.squeeze(np.mean(all_preds, axis=0))
    preds_ind = np.argmax(ens_preds, axis=1)
    ens_errors = (labels != preds_ind).astype("float32")

    thresh = 0.5
    # Get metric (R-AUC, F1-AUC, F1-95) for each uncertainty measure
    for measure in all_ens_uncertainty.keys():
        ens_uncertainties = all_ens_uncertainty[measure]
        rejection_mse = calc_uncertainty_regection_curve(ens_errors, ens_uncertainties)
        ens_r_auc = rejection_mse.mean()
        ens_f_auc, ens_f95, _ = f_beta_metrics(ens_errors, ens_uncertainties, thresh)

        single_r_aucs = []
        single_f_aucs = []
        single_f95s = []
        for seed in range(10):
            single_preds = np.expand_dims(all_preds[seed,:,:], axis=0)
            all_single_uncertainty = ensemble_uncertainties_classification(single_preds)
            single_uncertainties = all_single_uncertainty[measure]
            preds_ind = np.argmax(np.squeeze(single_preds), axis=1)
            single_errors = (labels != preds_ind).astype("float32")
            rejection_mse = calc_uncertainty_regection_curve(single_errors, single_uncertainties)
            single_r_aucs.append(rejection_mse.mean())
            single_f_auc, single_f95, _ = f_beta_metrics(single_errors, single_uncertainties, thresh)
            single_f_aucs.append(single_f_auc)
            single_f95s.append(single_f95)
        single_r_aucs = np.asarray(single_r_aucs)
        single_f_aucs = np.asarray(single_f_aucs)
        single_f95s = np.asarray(single_f95s)
        print()
        print(f'Measure {measure}')
        print(f'Ensemble R-AUC: {ens_r_auc}')
        print(f'Single R-AUC: {single_r_aucs.mean()} +- {single_r_aucs.std()}')
        print(f'Ensemble F1-AUC: {ens_f_auc}')
        print(f'Single F1-AUC: {single_f_aucs.mean()} +- {single_f_aucs.std()}')
        print(f'Ensemble F1@95: {ens_f95}')
        print(f'Single F1@95: {single_f95s.mean()} +- {single_f95s.std()}')
        print()
    print('----------------')