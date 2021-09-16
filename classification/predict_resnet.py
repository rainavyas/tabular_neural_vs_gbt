'''
Generate prediction files for mlp models in the following format:

Argument specifies destination directory in which this script
generates content with following contents:

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
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from train_resnet import get_lab_to_ind, ResNet, get_default_device

@torch.no_grad()
def get_predictions(dl, model, device):
    '''
    Returns predictions for specified model
    Classification: [num_samples x num_classes]
    '''
    model.eval()
    preds_list = []
    for (x,_) in dl:
        x = x.to(device)
        logits = model(x)
        s = nn.Softmax(dim=1)
        probs = s(logits)
        preds_list.append(probs)
    preds = torch.cat(preds_list).detach().cpu().numpy()
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='resnet predict.')
    parser.add_argument('trained_models_dir', type=str, help='Path to dir of trained models')
    parser.add_argument('out_dir', type=str, help='Path to dir to save prediction files')
    parser.add_argument('data_paths', type=str, help='list of all eval data files')


    args = parser.parse_args()
    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict_resnet.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    data_paths = args.data_paths.split()
    dfs = []
    for data_path in data_paths:
        dfs.append(pd.read_csv(data_path))
    df = pd.concat(dfs)

    print("Loaded Data")

    # 10 models assumed
    ensemble_size = 10

    # Get the device
    device = get_default_device()

    # Generate prediction file for each model
    for seed in range(1, ensemble_size+1):
        # Set Seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
        model = ResNet()
        model.load_state_dict(torch.load(f'{args.trained_models_dir}/seed{seed}.th', map_location='cpu'))
        model.to(device)

        X_dev_np = np.asarray(df.iloc[:,6:])
        X_dev = torch.FloatTensor(X_dev_np)

        lab_to_ind = get_lab_to_ind(df)
        batch_size = 512

        # get targets for dev
        y_dev = df['fact_cwsm_class']
        y_dev = torch.LongTensor(np.asarray([lab_to_ind[lab] for lab in y_dev]))
        labels = y_dev.detach().cpu().numpy()
        np.save(f'{args.out_dir}/targets.npy', labels)

        dev_ds = TensorDataset(X_dev, y_dev)
        dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

        # get prediction
        preds = np.asarray(get_predictions(dev_dl, model, device))
        np.save(f'{args.out_dir}/{seed}.npy', preds)

        print('Done seed', seed)