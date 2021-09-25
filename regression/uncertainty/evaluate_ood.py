'''
Peforms general OOD evaluation from a predictions file
(ensemble results)

Expects to be passed two directories (in and out domain) with following contents each:

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

import argparse
import os
import sys
from uncertainty import ensemble_uncertainties_regression
from assessment import ood_detect
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval regression')
    parser.add_argument('preds_indir', type=str, help='Path to dir of indomain predictions')
    parser.add_argument('preds_outdir', type=str, help='Path to dir of outdomain predictions')

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate_ood.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')


    # Load preds
    all_means = []
    all_variances = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_means.append(np.load(f'{args.preds_indir}/mean{seed}.npy'))
        all_variances.append(np.load(f'{args.preds_indir}/variance{seed}.npy'))
    
    # Convert to array: [num_models, num_examples, 2]
    all_means = np.stack(all_means)
    all_variances = np.stack(all_variances)
    all_preds_in = np.stack((all_means, all_variances), axis=-1)

    all_means = []
    all_variances = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_means.append(np.load(f'{args.preds_outdir}/mean{seed}.npy'))
        all_variances.append(np.load(f'{args.preds_outdir}/variance{seed}.npy'))
    
    # Convert to array: [num_models, num_examples, 2]
    all_means = np.stack(all_means)
    all_variances = np.stack(all_variances)
    all_preds_out = np.stack((all_means, all_variances), axis=-1)

    domain_labels = np.asarray([0]*(all_preds_in.shape[1]) + [1]*(all_preds_out.shape[1]))

    all_ens_uncertainty_in = ensemble_uncertainties_regression(all_preds_in)
    all_ens_uncertainty_out = ensemble_uncertainties_regression(all_preds_out)

    # Get metric (ROC-AUC) for each uncertainty measure
    for measure in all_ens_uncertainty_in.keys():
        ens_uncertainties_in = all_ens_uncertainty_in[measure]
        ens_uncertainties_out = all_ens_uncertainty_out[measure]

        ens_ood_auc = ood_detect(domain_labels, ens_uncertainties_in, ens_uncertainties_out)
        print(measure)
        print("Ensemble")
        print("ROC-AUC:", ens_ood_auc)
        print()