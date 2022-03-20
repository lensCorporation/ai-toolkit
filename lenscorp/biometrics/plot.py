"""PLOTTING UTILITY FOR BIOMETRIC APPLICATIONS
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

def roc_multiple(benchmarks, lim_x=None, filename='roc.png'):
    """Plot multiple ROC plots in a single plot

    Args:
        tars (_type_):              _description_
        fars (_type_):              _description_
        filename (str, optional):   Filename to save plot. 
                                    Defaults to 'roc.png'.
    """
    
    min_y = []
    for item in benchmarks.keys():
        tars = benchmarks[item]['tars']
        fars = benchmarks[item]['fars']
        if utils.is_list(tars) and utils.is_list(fars):
            tars = np.array(tars)
            fars = np.array(fars)
        if tars.max() <= 1.:
            print('Converting to percentage')
            tars = tars * 100.
            fars = fars * 100.
        if lim_x is not None:
            assert lim_x[0] < lim_x[1]
            for j, val in enumerate(fars):
                if fars[j] > lim_x[0]:
                    min_y.append(tars[j])
                    break
        sns.lineplot(fars, tars, label=item)
    plt.xlabel('False Accept Rate (%)')
    plt.ylabel('True Accept Rate (%)')
    plt.xscale('log')
    if lim_x is not None:
        plt.xlim([lim_x[0], lim_x[1]])
        plt.ylim([min(min_y), 100.])
    plt.savefig(filename)
    plt.clf()
    plt.close()
    

def roc(tars, fars, filename='roc.png'):
    """Plot single ROC plot

    Args:
        tars (_type_):              _description_
        fars (_type_):              _description_
        filename (str, optional):   Filename to save plot. 
                                    Defaults to 'roc.png'.
    """
    if utils.is_list(tars) and utils.is_list(fars):
        tars = np.array(tars)
        fars = np.array(fars)
    if tars.max() <= 1.:
        print('Converting to percentage')
        tars = tars * 100.
        fars = fars * 100.
        
    sns.lineplot(fars, tars)
    plt.xlabel('False Accept Rate (%)')
    plt.ylabel('True Accept Rate (%)')
    plt.xscale('log')
    plt.savefig(filename)
    plt.clf()
    plt.close()