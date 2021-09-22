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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
import argparse
import os
import sys
import matplotlib.pyplot as plt


def classification_calibration(labels, probs, bins=10, save_path=None):
    n_classes = np.float(probs.shape[-1])
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
        acc = np.mean(np.asarray(llabels == lpreds, dtype=np.float32))
        prob = np.mean(lprobs)
        if np.isnan(acc):
            acc = 0.0
            prob = 0.0
        ECE += np.abs(acc - prob) * float(lprobs.shape[0])
        gaps[i] = np.abs(acc - prob)
        accs[i] = acc
        upper += increment
        lower += increment
    ECE /= np.float(total)
    MCE = np.max(np.abs(gaps))
    accs[-1] = 1.0
    return confs, accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval classification')
    parser.add_argument('preds_dir1', type=str, help='Path to dir of predictions')
    parser.add_argument('preds_dir2', type=str, help='Path to dir of predictions')
    parser.add_argument('preds_dir3', type=str, help='Path to dir of predictions')
    parser.add_argument('preds_dir4', type=str, help='Path to dir of predictions')
    parser.add_argument('save_path', type=str, help='Path to save figures')

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/curves.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # labels
    labels = np.load(f'{args.preds_dir1}/targets.npy')

    # Load preds
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_dir1}/{seed}.npy'))
    
    # Ensemble results
    ens_preds = np.mean(np.stack(all_preds), axis=0)

    confs1, accs1 = classification_calibration(labels, ens_preds, save_path=args.save_path)

    # labels
    labels = np.load(f'{args.preds_dir2}/targets.npy')

    # Load preds
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_dir2}/{seed}.npy'))

    # Ensemble results
    ens_preds = np.mean(np.stack(all_preds), axis=0)

    confs2, accs2 = classification_calibration(labels, ens_preds, save_path=args.save_path)

    # labels
    labels = np.load(f'{args.preds_dir2}/targets.npy')

    # Load preds
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_dir3}/{seed}.npy'))

    # Ensemble results
    ens_preds = np.mean(np.stack(all_preds), axis=0)

    confs3, accs3 = classification_calibration(labels, ens_preds, save_path=args.save_path)

    # labels
    labels = np.load(f'{args.preds_dir2}/targets.npy')

    # Load preds
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_dir4}/{seed}.npy'))

    # Ensemble results
    ens_preds = np.mean(np.stack(all_preds), axis=0)

    confs4, accs4 = classification_calibration(labels, ens_preds, save_path=args.save_path)


    n_classes = 9 
    fig, ax = plt.subplots(dpi=300)
    plt.plot(confs1, accs1)
    plt.plot(confs2, accs2)
    plt.plot(confs3, accs3)
    plt.plot(confs4, accs4)
    plt.plot(confs1, confs1)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.xlim(1.0/n_classes, 1.0)
    plt.legend(['MLP', 'ResNet', 'XGBoost', 'CatBoost', 'Ideal'])
    plt.savefig(args.save_path, bbox_inches='tight')
    plt.close()
