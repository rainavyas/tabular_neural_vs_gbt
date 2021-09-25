'''
Peforms general classification evaluation from a predictions file
(single seed and ensemble results)

Expects to be passed two directories (in and out) each with following contents:

- targets.npy
- 1.npy
- 2.npy
 .
 .
 .
- 10.npy

where each seed prediction file contains a numpy array: [num_samples x num_classes]
'''

import argparse
import os
import sys
from uncertainty import ensemble_uncertainties_classification
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

    # Load preds in
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_indir}/{seed}.npy'))
    all_preds_in = np.stack(all_preds, axis=0)

    # Load preds out
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_outdir}/{seed}.npy'))
    all_preds_out = np.stack(all_preds, axis=0)

    domain_labels = np.asarray([0]*(all_preds_in.shape[1]) + [1]*(all_preds_out.shape[1]))

    all_ens_uncertainty_in = ensemble_uncertainties_classification(all_preds_in)
    all_ens_uncertainty_out = ensemble_uncertainties_classification(all_preds_out)

    # Get metric (ROC-AUC) for each uncertainty measure
    for measure in all_ens_uncertainty_in.keys():
        ens_uncertainties_in = all_ens_uncertainty_in[measure]
        ens_uncertainties_out = all_ens_uncertainty_out[measure]

        ens_ood_auc = ood_detect(domain_labels, ens_uncertainties_in, ens_uncertainties_out)
        print(measure)
        print("Ensemble")
        print("ROC-AUC:", ens_ood_auc)
        print()