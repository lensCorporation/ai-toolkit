"""EVALUATION METRICS FOR BIOMETRIC APPLICATIONS
"""

import numpy as np
from logging import warning
from tqdm import tqdm

def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=10e-8):
    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        epsilon = 10e-5
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = (num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds

def accuracy(score_vec, label_vec, thresholds=None):
    """ Compute Binary Classification Accuracy (Predicting whether two features belong to same person or not)

    Args:
        score_vec (array of float):             Vector containing scores
        label_vec (bool):                       Vector containing labels (0 : different people, 1 : same person)
        thresholds (float, optional):           Pre-defined threshold for computing accuracy.
                                                Defaults to None.
    Returns:
        accuracy: _description_
        threshold: _description_
    """
    assert len(score_vec.shape)==1
    assert len(label_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype==np.bool
    
    if thresholds is None:
        score_pos = score_vec[label_vec==True]
        thresholds = np.sort(score_pos)[::1]    

    assert len(thresholds.shape)==1
    if np.size(thresholds) > 10000:
        warning('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))
    
    accuracies = np.zeros(np.size(thresholds))
    for i, threshold in enumerate(thresholds):
        pred_vec = score_vec>=threshold
        accuracies[i] = np.mean(pred_vec==label_vec)
        
    argmax = np.argmax(accuracies)
    accuracy = accuracies[argmax]
    threshold = np.mean(thresholds[accuracies==accuracy])

    return accuracy, threshold


def ROC(score_vec, label_vec, thresholds=None, FARs=None, get_false_indices=False):
    """ Compute vectorized Receiver Operating Characteristic (ROC) curve.

    Args:
        score_vec (array of float):             Vector containing scores
        label_vec (bool):                       Vector containing labels (0 : different people, 1 : same person)
        thresholds (float, optional):           Pre-defined threshold. Defaults to None.
        FARs (list of float, optional):         True accept rates at specific False Accept Rates. 
                                                Defaults to None.
        get_false_indices (bool, optional):     Return misclassification errors.
                                                Defaults to False.

    Returns:
        TARs:                                   list of true accept rate values at certain False Accept Rates
        FARs:                                   list of false accept rate values where true accept rates are computed
        thresholds:                             list of threshold values at certain False Accept Rates
        false_accept_indices (optional):        indices where false accepts occur
        false_reject_indices (optional):        list of indices where false rejects occur
    """
    assert score_vec.ndim == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    
    if thresholds is None:
        thresholds = find_thresholds_by_FAR(score_vec, label_vec, FARs=FARs)

    assert len(thresholds.shape)==1 
    if np.size(thresholds) > 10000:
        warning('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))

    # FARs would be check again
    TARs = np.zeros(thresholds.shape[0])
    FARs = np.zeros(thresholds.shape[0])
    false_accept_indices = []
    false_reject_indices = []
    for i,threshold in enumerate(thresholds):
        accept = score_vec >= threshold
        TARs[i] = np.mean(accept[label_vec])
        FARs[i] = np.mean(accept[~label_vec])
        if get_false_indices:
            false_accept_indices.append(np.argwhere(accept & (~label_vec)).flatten())
            false_reject_indices.append(np.argwhere((~accept) & label_vec).flatten())

    if get_false_indices:
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds


def ROC_by_mat(score_mat, label_mat, thresholds=None, FARs=None, get_false_indices=False, triu_k=None):
    """ Compute matrix-based Receiver Operating Characteristic (ROC) curve.

    Args:
        score_mat (2-D matrix of float):        P x G matrix, P is number of probes, G is size of gallery
        label_mat (bool):                       P x G matrix
        thresholds (float, optional):           Pre-defined threshold. Defaults to None.
        FARs (list of float, optional):         True accept rates at specific False Accept Rates. 
                                                Defaults to None.
        get_false_indices (bool, optional):     Return misclassification errors.
                                                Defaults to False.
        triu_k:                                 Offset from diagonal (k = 1 implies upper triangular matrix without diagonal).
                                                Defaults to None.

    Returns:
        TARs:                                   list of true accept rate values at certain False Accept Rates
        FARs:                                   list of false accept rate values where true accept rates are computed
        thresholds:                             list of threshold values at certain False Accept Rates
        false_accept_indices (optional):        list of indices where false accepts occur
        false_reject_indices (optional):        list of indices where false rejects occur
    """
    assert score_mat.ndim == 2
    assert score_mat.shape == label_mat.shape
    assert label_mat.dtype == np.bool
    
    # Convert into vectors
    m,n  = score_mat.shape
    if triu_k is not None:
        assert m==n, "If using triu for ROC, the score matrix must be a sqaure matrix!"
        triu_indices = np.triu_indices(m, triu_k)
        score_vec = score_mat[triu_indices]
        label_vec = label_mat[triu_indices]
    else:
        score_vec = score_mat.flatten()
        label_vec = label_mat.flatten()

    # Compute ROC
    if get_false_indices:
        TARs, FARs, thresholds, false_accept_indices, false_reject_indices = \
                    ROC(score_vec, label_vec, thresholds, FARs, True)
    else:
        TARs, FARs, thresholds = ROC(score_vec, label_vec, thresholds, FARs, False)

    # Convert false accept/reject indices into [row, col] indices
    if get_false_indices:
        rows, cols = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
        rc = np.stack([rows, cols], axis=2)
        if triu_k is not None:
            rc = rc[triu_indices,:]
        else:
            rc = rc.reshape([-1,2])

        for i in range(len(FARs)):
            false_accept_indices[i] = rc[false_accept_indices[i]]
            false_reject_indices[i] = rc[false_reject_indices[i]]
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds
    

def Full_ROC_by_mat(score_mat, k=100, FARs=None):
    """ Estimate entire matrix-based Receiver Operating Characteristic (ROC) curve.
        NOTE: Currently assumes probes and gallery are ordered such that:
            (1) Single probe and gallery mate pairs
            (2) Probe and Gallery mates are always present ONLY in the diagonals
        
    Args:
        score_mat (2-D matrix of float):        P x G matrix, P is number of probes, G is size of gallery
        k (int):                                Top-k ranks to consider for high-impostor (FAR calculations)
        FARs (list of float, optional):         True accept rates at specific False Accept Rates. 
                                                Defaults to None.

    Returns:
        TARs:                                   list of true accept rate values at certain False Accept Rates
        FARs:                                   list of false accept rate values where true accept rates are computed
        thresholds:                             list of threshold values at certain False Accept Rates
    """
    assert score_mat.ndim == 2
    
    num_probes = score_mat.shape[0]
    gen = np.empty((num_probes, 1), dtype=np.float32)
    imp_non_zero = np.empty((num_probes * (k-1), ), dtype=np.float32)
    num_imp_zeros = len(gen) * (len(gen) - k)
    
    for probe in tqdm(range(num_probes)):
        # Extract top-K candidates (highest scores)
        idx = np.argpartition(score_mat[probe], -k)[-k:]
        highest_score_index = idx[score_mat[probe][idx].argmax()]
        # Highest score on diagonal (correct mate at first rank)
        if highest_score_index == probe:
            gen[probe] = score_mat[probe, highest_score_index]
        else:
            gen[probe] = 0.
        # Remaining (K - 1) scores
        remaining_score_idx = np.where(~np.in1d(idx, highest_score_index))[0]
        imp_non_zero[probe * (k - 1) : (probe + 1) * (k - 1)] = score_mat[probe, remaining_score_idx]
    
    frrs = []
    fars = []
    thresholds = []
    for score in tqdm(range(2001)):
        threshold = score/2000
        frrs.append(len(np.where(gen <= threshold)[0]) / len(gen))
        fars.append(len(np.where(imp_non_zero > threshold)[0])  / (len(imp_non_zero) + num_imp_zeros))
        thresholds.append(threshold)
        
    sort_idx = np.argsort(fars)
    fars = np.array(fars)[sort_idx]
    tars = 1. - np.array(frrs)[sort_idx]
    thresholds = np.array(thresholds)[sort_idx]
    
    if FARs is not None:
        TARs_to_return = []
        FARs_to_return = []
        thresholds_to_return = []
        for FAR in FARs:
            for j in range(len(fars)):
                if fars[j] > FAR:
                    TARs_to_return.append(tars[j])
                    FARs_to_return.append(fars[j])
                    thresholds_to_return.append(thresholds[j])
                    break
        return TARs_to_return, FARs_to_return, thresholds_to_return
    else:
        return tars, fars, thresholds


def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0]):
    """ Closed/Open-set Identification.
        A general case of Cummulative Match Characteristic (CMC) 
        where thresholding is allowed for open-set identification.

    Args:
        score_mat (2-D matrix of float):        P x G matrix, P is number of probes, G is size of gallery
        label_mat (bool):                       P x G matrix
        ranks (list, optional):                 list of integers. Defaults to [1].
        FARs (list, optional):                  false alarm rates, if 1.0, closed-set identification (CMC).
                                                Defaults to [1.0].

    Returns:
        DIRs:                                   F x R matrix, F is the number of FARs, R is the number of ranks. 
                                                Flatten into a vector if F=1 or R=1.
        FARs:                                   Vector of length = F.
        thresholds:                             Vector of length = F.
    """
    assert score_mat.shape==label_mat.shape
    assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    
    match_indices = label_mat.astype(np.bool).any(axis=1)
    score_mat_m = score_mat[match_indices,:]
    label_mat_m = label_mat[match_indices,:]
    score_mat_nm = score_mat[np.logical_not(match_indices),:]

    print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))
    
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=np.bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        thresholds = [np.min(score_mat) - 1e-10]
    else:
        assert score_mat_nm.shape[0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)
        
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=np.bool)
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])[::-1]
        sorted_label_mat_m[row,:] = label_mat_m[row, sort_idx]
        
    gt_score_m = score_mat_m[label_mat_m]
    assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    for i, threshold in enumerate(thresholds):
        for j, rank  in enumerate(ranks):
            score_rank = gt_score_m >= threshold
            retrieval_rank = sorted_label_mat_m[:,0:rank].any(axis=1)
            DIRs[i,j] = (score_rank & retrieval_rank).astype(np.float32).mean()
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()

    return DIRs, FARs, thresholds


