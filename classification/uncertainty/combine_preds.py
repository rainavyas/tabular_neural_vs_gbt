'''
Given two different dirs of predictions, combine
into a single dir by concatenation
'''

import numpy as np
import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combine_preds')
    parser.add_argument('dir1', type=str, help='Path to input dir of predictions1')
    parser.add_argument('dir2', type=str, help='Path to input dir of predictions2')
    parser.add_argument('out_dir', type=str, help='Path to output predictions dir')

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/combine_preds.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    ensemble_size = 10

    labels1 = np.load(f'{args.dir1}/targets.npy')
    labels2 = np.load(f'{args.dir2}/targets.npy')
    labels = np.concatenate((labels1, labels2), axis=0)
    np.save(f'{args.out_dir}/targets.npy', labels)

    for seed in range(1, ensemble_size+1):
        preds1 = np.load(f'{args.dir1}/{seed}.npy')
        preds2 = np.load(f'{args.dir2}/{seed}.npy')
        preds = np.concatenate((preds1, preds2), axis=0)
        np.save(f'{args.out_dir}/{seed}.npy', preds)
        print('Done seed', seed)
