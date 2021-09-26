'''
Expects classification predictions dir at the input

Generates following plots:
1) Error retention curve (single and ensemble, reporting r-AUC)
2) F1 retention curve (single and ensemble, reporting F1-AUC)

This is done for the confidence uncertainty measure
'''

import numpy as np
import argparse
import os
import sys
import math
from uncertainty import ensemble_uncertainties_classification
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
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_dir}/{seed}.npy'))
    all_preds = np.stack(all_preds, axis=0)

    # get all uncertainties
    all_ens_uncertainty = ensemble_uncertainties_classification(all_preds)
    all_ens_uncertainty['confidence'] = all_ens_uncertainty['confidence'] * (-1.0)

    

    # Get errors
    measure = 'confidence'
    ens_preds = np.squeeze(np.mean(all_preds, axis=0))
    preds_ind = np.argmax(ens_preds, axis=1)
    ens_errors = (labels != preds_ind).astype("float32")
    ens_uncertainties = all_ens_uncertainty[measure]

    # Repeat for single seed - choose seed 0
    measure = 'entropy_of_expected'
    single_preds = np.expand_dims(all_preds[0,:,:], axis=0)
    all_single_uncertainty = ensemble_uncertainties_classification(single_preds)
    single_uncertainties = all_single_uncertainty[measure]
    preds_ind = np.argmax(np.squeeze(single_preds), axis=1)
    single_errors = (labels != preds_ind).astype("float32")

    # Error retention curve

    # Ensemble
    rejection_error = calc_uncertainty_regection_curve(ens_errors, ens_uncertainties)
    ens_r_auc = rejection_error.mean()
    ens_retention_error = rejection_error[::-1]

    # Single
    rejection_error = calc_uncertainty_regection_curve(single_errors, single_uncertainties)
    single_r_auc = rejection_error.mean()
    single_retention_error = rejection_error[::-1]

    retention_fractions = np.linspace(0,1,len(ens_retention_error))

    # Plot
    data_ens = pd.DataFrame(data={'x':retention_fractions, 'y':ens_retention_error})
    data_sin = pd.DataFrame(data={'x':retention_fractions, 'y':single_retention_error})

    filename = f'{args.out_prefix}_retention_error.png'
    sns.set(rc={'figure.figsize':(5,3.3)})
    sns.lineplot(data=data_sin, x='x', y='y', label=f'Single R-AUC: {single_r_auc:.3}')
    sns.lineplot(data=data_ens, x='x', y='y', label=f'Ensemble R-AUC: {ens_r_auc:.3}')
    plt.ylabel('Error')
    plt.xlabel("Retention Fraction")
    plt.xlim([-0.01,1.01])
    plt.ylim(bottom=-0.01)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.clf()



    # F1 Retention plot
    thresh = 0.5

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
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.clf()