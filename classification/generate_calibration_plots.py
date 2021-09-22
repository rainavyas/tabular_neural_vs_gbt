'''
Generates calibration diagrams for the specified dataset:
 - Single Seed
 - Ensemble

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

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def classification_calibration(labels, probs, bins=10):
    lower = 0
    preds = np.argmax(probs, axis=1)
    total = labels.shape[0]
    probs = np.max(probs, axis=1)
    increment = 1.0 / bins
    upper = increment
    accs = np.zeros([bins + 1], dtype=np.float32)
    gaps = np.zeros([bins + 1], dtype=np.float32)
    confs = np.arange(0, 1.01, increment)
    ECE = 0.0
    for i in range(bins):
        ind1 = probs >= lower
        ind2 = probs < upper
        ind = np.where(np.logical_and(ind1, ind2))[0]
        lprobs = probs[ind]
        lpreds = preds[ind]
        llabels = labels[ind]
        try:
            acc = np.mean(np.asarray(llabels == lpreds, dtype=np.float32))
            prob = np.mean(lprobs)
        except:
        # if np.isnan(acc):
            acc = 0.0
            prob = 0.0
        ECE += np.abs(acc - prob) * float(lprobs.shape[0])
        gaps[i] = np.abs(acc - prob)
        accs[i] = acc
        upper += increment
        lower += increment
    ECE /= float(total)
    MCE = np.max(np.abs(gaps))
    accs[-1] = 1.0
    return confs, accs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate plots')
    parser.add_argument('preds_dir', type=str, help='Path to dir of predictions')
    parser.add_argument('out_prefix', type=str, help='Path to dir to save plots and prefix')

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/generate_calibration_plots.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # labels
    labels = np.load(f'{args.preds_dir}/targets.npy')

    # Load preds
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_dir}/{seed}.npy'))
    
    # preds
    print("Getting preds")
    ens_preds = np.mean(np.stack(all_preds), axis=0)
    single_preds = all_preds[0]

    print("Plotting")
    # Ensemble plot
    ens_confs, ens_accs = classification_calibration(labels, ens_preds)
    plt.bar(ens_confs, ens_accs, width=1.0)
    plt.plot(ens_confs, ens_confs)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.savefig(f'{args.out_prefix}_ensemble.png')


