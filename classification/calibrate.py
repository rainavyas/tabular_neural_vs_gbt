import numpy as np
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
    if save_path is not None:
        fig, ax = plt.subplots(dpi=300)
        plt.plot(confs, accs)
        plt.plot(confs, confs)
        plt.ylim(0.0, 1.0)
        plt.ylabel('Accuracy')
        plt.xlabel('Confidence')
        plt.xlim(1.0/n_classes, 1.0)
        plt.legend(['Model','Ideal'])
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    return np.round(ECE * 100.0, 2), np.round(MCE * 100.0, 2)

def get_macro_nll(labels, probs):
    '''
    Aeverage Negative-log likelihood per class,
    then averaged over classes
    '''
    num_classes = probs.shape[-1]
    likelihoods = probs[labels]
    nll_by_class = []
    for i in range(num_classes):
        relevant_likelihoods = likelihoods[labels[:]==i]
        nll_this_class = np.mean(-np.log(relevant_likelihoods))
        if nll_this_class.isnan():
            nll_this_class = 0
        nll_by_class.append(nll_this_class)
    return np.mean(np.asarray(nll_by_class))

def eval_calibration(labels, probs, save_path=None, bins=10):
    likelihoods = probs[labels]
    nll = np.mean(-np.log(likelihoods))
    macro_nll = get_macro_nll(labels, probs)
    brier = np.mean((1.0-likelihoods)**2)

    ece, mce = classification_calibration(labels,probs, bins=bins, save_path=save_path)
    return np.round(nll,2), np.round(brier,2), np.round(ece,2), np.round(mce,2), np.round(macro_nll,2)



