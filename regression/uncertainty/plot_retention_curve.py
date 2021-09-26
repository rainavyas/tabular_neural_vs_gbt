'''
Expects regression predictions dir at the input

Generates following plots:
1) MSE retention curve (single and ensemble, reporting r-AUC)
2) F1 retention curve (single and ensemble, reporting F1-AUC)

This is done for the tvar uncertainty measure
'''

import numpy as np
import argparse
import os
import sys
import math
from uncertainty import ensemble_uncertainties_regression
from assessment import calc_uncertainty_regection_curve, f_beta_metrics
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval regression')
    parser.add_argument('preds_dir', type=str, help='Path to dir of predictions')
    parser.add_argument('out_prefix', type=str, help='Prefix for output figures saved')


    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/plot_retention_curve.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Set plot style
    plt.style.use('seaborn')
    matplotlib.rcParams.update({
        "axes.labelsize": 18,
        "font.size": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.frameon": True
    })
    
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

    measure = 'tvar'

    # Get errors
    all_preds_mean = all_preds[:,:,0]
    avg_preds = np.squeeze(np.mean(all_preds_mean, axis=0))
    ens_errors = (avg_preds - labels) ** 2
    ens_uncertainties = all_ens_uncertainty[measure]

    # Repeat for single seed - choose seed 0
    single_preds = np.expand_dims(all_preds[0,:,:], axis=0)
    all_single_uncertainty = ensemble_uncertainties_regression(single_preds)
    single_uncertainties = all_single_uncertainty[measure]
    single_preds_mean = single_preds[:,:,0]
    single_avg_preds = np.squeeze(np.mean(single_preds_mean, axis=0))
    single_errors = (single_avg_preds - labels) ** 2



    # MSE retention curve

    # Ensemble
    rejection_mse = calc_uncertainty_regection_curve(ens_errors, ens_uncertainties)
    ens_r_auc = rejection_mse.mean()
    ens_retention_mse = rejection_mse[::-1]

    # Single
    rejection_mse = calc_uncertainty_regection_curve(single_errors, single_uncertainties)
    single_r_auc = rejection_mse.mean()
    single_retention_mse = rejection_mse[::-1]

    retention_fractions = np.linspace(0,1,len(ens_retention_mse))

    # Plot
    data_ens = pd.DataFrame(data={'x':retention_fractions, 'y':ens_retention_mse})
    data_sin = pd.DataFrame(data={'x':retention_fractions, 'y':single_retention_mse})

    filename = f'{args.out_prefix}_retention_mse.png'
    sns.set(rc={'figure.figsize':(5,3.3)})
    sns.lineplot(data=data_sin, x='x', y='y', label=f'Single R-AUC: {single_r_auc:.3}')
    sns.lineplot(data=data_ens, x='x', y='y', label=f'Ensemble R-AUC: {ens_r_auc:.3}')
    plt.ylabel('MSE')
    plt.xlabel("Retention Fraction")
    plt.xlim([-0.01,1.01])
    plt.ylim(bottom=-0.01)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.clf()



    # F1 Retention plot
    thresh = 1.0

    ens_f_auc, _, ens_retention_f1 = f_beta_metrics(ens_errors, ens_uncertainties, thresh)
    single_f_auc, _, single_retention_f1 = f_beta_metrics(single_errors, single_uncertainties, thresh)
    retention_fractions = np.linspace(0,1,len(ens_retention_f1))

    # Plot
    data_ens = pd.DataFrame(data={'x':retention_fractions, 'y':ens_retention_f1})
    data_sin = pd.DataFrame(data={'x':retention_fractions, 'y':single_retention_f1})

    filename = f'{args.out_prefix}_retention_f1.png'
    sns.set(rc={'figure.figsize':(5,3.3)})
    sns.lineplot(data=data_sin, x='x', y='y', label=f'Single F1-AUC: {single_f_auc:.3}')
    sns.lineplot(data=data_ens, x='x', y='y', label=f'Ensemble F1-AUC: {ens_f_auc:.3}')
    plt.ylabel('F1 Score')
    plt.xlabel("Retention Fraction")
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()