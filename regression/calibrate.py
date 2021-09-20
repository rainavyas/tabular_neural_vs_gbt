import numpy as np
import math

def gaussian_negative_log_likelihood(means, variances, targets):
    nlls =  0.5 * ( np.log(variances) + np.divide((targets-means)**2, variances) + np.log(2*math.pi) )
    return nlls.mean()
