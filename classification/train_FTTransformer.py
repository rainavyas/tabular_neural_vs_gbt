import rtdl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scipy import stats
import numpy as np



def apply_model(model, x_num, x_cat=None):
    '''
    FTTransformer expects numerical and categorical inputs separately
    '''
    return model(x_num, x_cat) if isinstance(model, rtdl.FTTransformer) else model(x_num)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
      curr_ind = pred[:,curr_k]
      num_eq = torch.eq(curr_ind, target).sum()
      acc = num_eq/len(output)
      res_total += acc
    return res_total*100

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=2000):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = apply_model(model, x)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, prec=accs))

@torch.no_grad()
def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()


    for i, (x, target) in enumerate(val_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        logits = apply_model(model, x)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

    print('Dev in\t Loss ({loss.avg:.4f})\t'
            'Accuracy ({prec.avg:.3f})\n'.format(
              loss=losses, prec=accs))


def get_lab_to_ind(data_df):
    '''
    Prepare a label to index map
    '''
    y_fact = set(list(data_df['fact_cwsm_class']))
    lab_to_ind = {}
    for i, lab in enumerate(y_fact):
        lab_to_ind[lab] = i
    return lab_to_ind

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def main():

    parser = argparse.ArgumentParser(description='Train FTTransformer.')
    parser.add_argument('train_path', type=str, help='Path to train data')
    parser.add_argument('dev_in_path', type=str, help='Path to dev_in data')
    parser.add_argument('--epochs', type=int, default=50, help='Specify the number of epochs to train for')
    parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--save_dir', type=str, help='Load path to which trained model will be saved')

    args = parser.parse_args()

    df_train = pd.read_csv(args.train_path)
    df_dev_in = pd.read_csv(args.dev_in_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    seed = args.seed

    lab_to_ind = get_lab_to_ind(df_train)
    batch_size = args.batch_size

    X_train_np = np.asarray(df_train.iloc[:,6:])
    X_dev_in_np = np.asarray(df_dev_in.iloc[:,6:])

    X_train = torch.FloatTensor(X_train_np)
    X_dev_in = torch.FloatTensor(X_dev_in_np)


    # Train
    y_train = np.asarray(df_train['fact_cwsm_class'])
    y_train = torch.LongTensor(np.asarray([lab_to_ind[lab] for lab in y_train]))

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Dev in
    y_dev_in = df_dev_in['fact_cwsm_class']
    y_dev_in = torch.LongTensor(np.asarray([lab_to_ind[lab] for lab in y_dev_in]))

    dev_in_ds = TensorDataset(X_dev_in, y_dev_in)
    dev_in_dl = DataLoader(dev_in_ds, batch_size=batch_size, shuffle=True)


    device = get_default_device()

    # Create the Feature Transformer Model

    model = rtdl.FTTransformer.make_default(
        n_num_features=X_train.shape[1],
        cat_cardinalities=None,
        last_layer_query_idx=[-1],
        d_out=len(lab_to_ind)
    )

    model.to(device)

    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    )



    # create loss function criterion
    criterion = nn.CrossEntropyLoss().to(device)


    # Train
    epochs = args.epochs
    for epoch in range(epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        eval(dev_in_dl, model, criterion, device)




    state = model.state_dict()
    torch.save(state, f'{args.save_dir}/model{args.seed}.th')




if __name__ == '__main__':
    main()
