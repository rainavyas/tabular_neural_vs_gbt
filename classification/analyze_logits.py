'''
Analyzes logit size (pre-softmax)

Expects to be passed a directory with following contents:

- targets.npy
- logits1.npy
- logits2.npy
.
.
.
- logits10.npy
where each logits file is a numpy array: [num_samples x num_classes]

'''

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval classification')
    parser.add_argument('in_preds_dir', type=str, help='Path to dir of in-domain predictions')
    parser.add_argument('out_preds_dir', type=str, help='Path to dir of out-domain predictions')
    parser.add_argument('out_file', type=str, help='output .png histogram')

    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analyze_logits.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load logits - use only seed 1
    in_domain_logits = np.load(f'{args.in_preds_dir}/logits1.npy')
    out_domain_logits = np.load(f'{args.out_preds_dir}/logits1.npy')

    # Select largest logits per sample
    in_domain_logits = np.max(in_domain_logits, axis=1)
    out_domain_logits = np.max(in_domain_logits, axis=1)

    # Histogram distribution
    _, bins, _ = plt.hist(in_domain_logits, bins=50, density=True, label='in domain')
    _ = plt.hist(out_domain_logits, bins=bins, alpha=0.5, density=True, label='out domain')
    plt.legend()
    plt.savefig(args.out_file)