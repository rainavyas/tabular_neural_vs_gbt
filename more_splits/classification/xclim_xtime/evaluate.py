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
from calibrate import eval_calibration

def metric_accuracy(preds, targets):
    pred_inds = np.argmax(np.asarray(preds), axis=1)
    return accuracy_score(targets, pred_inds)

def get_avg_f1(preds, labels):
    '''
    Calculate one-vs-all f1 score per class
    Return average of f1 scores over all classes
    preds: [num_samples x num_classes]
    '''
    preds = preds.tolist()
    labels = labels.tolist()
    f1s = []
    class_inds_to_check = list(set(labels))

    for class_ind_to_check in class_inds_to_check:
        y_true = []
        y_pred = []
        for pred, lab_ind in zip(preds, labels):
            y_pred.append(pred[class_ind_to_check])
            if lab_ind == class_ind_to_check:
                y_true.append(1)
            else:
                y_true.append(0)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        f_scores = (2*precision*recall)/(precision+recall)
        f_scores_clean = f_scores[np.logical_not(np.isnan(f_scores))]
        f1s.append(np.amax(f_scores_clean))
    return np.mean(np.asarray(f1s))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval classification')
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
    all_preds = []
    ensemble_size = 10
    for seed in range(1, ensemble_size+1):
        all_preds.append(np.load(f'{args.preds_dir}/{seed}.npy'))
    
    # Ensemble results
    ens_preds = np.mean(np.stack(all_preds), axis=0)
    acc = metric_accuracy(ens_preds, labels)
    f1 = get_avg_f1(ens_preds, labels)
    print('---------------')
    print('Ens')
    print('Accuracy:', acc)
    print('Macro F1:', f1)
    print('---------------')

    # Single seed results
    accs = []
    f1s = []
    for preds in all_preds:
        accs.append(metric_accuracy(preds, labels))
        f1s.append(get_avg_f1(preds, labels))
    accs = np.asarray(accs)
    f1s = np.asarray(f1s)
    print('---------------')
    print('Single')
    print(f'Accuracy: {accs.mean()} +- {accs.std()}')
    print(f'Macro F1: {f1s.mean()} +- {f1s.std()}')
    print('---------------')
    print('---------------')
