"""
Utility functions for glhmm.
Written by: Joey Zhou
"""

import numpy as np
from glhmm import utils

#%% Translated from `HMM-MAR/task/utils/continuous_prediction_2class.m`
def continuous_prediction_2class(Y, Ypred):
    '''which one is the most likely class, out of a continuous estimation?
    INPUT:
        Y    : array of size (n_timepoints, n_stimfeature)
        Ypred: array of size (n_timepoints, n_stimfeature)

    OUTPUT:
        Ypred: array of ones and zeros of size (n_timepoints, n_stimfeature)
    '''

    N, q = Y.shape  # N is time points
    L = len(np.unique(Y)) 
    if q == 1 and L == 2:  # class is -1 or +1
        Ypred[Ypred <= 0] = -1
        Ypred[Ypred > 0] = 1
    elif q > 1 and L == 2:  # dummy variable representation for classification
        m = np.argmax(Ypred, axis=1)
        Ypred = np.zeros_like(Ypred)
        ind = np.arange(N), m
        Ypred[ind] = 1 
    return Ypred


#%% Model prediction based on state activation time course
def get_predicted(hmm, Gamma, X, Y=None):
    '''
    INPUT:
        hmm  : the trained glhmm.glhmm.hmm object
        Gamma: the estimated state time course of size (n_timepoints, n_states)
        X    : the time series data, array-like of shape (n_timepoints, n_parcels)
    
    OUTPUT:
        Ypred: predicted stimulus per time point
            array of ones and zeros of size (n_timepoints, n_stimefeature)
    '''
    
    #get predicted Y label
    B = hmm.get_betas()
    n_ch = B.shape[0] #number of channels/PCs
    n_sf = B.shape[1] #number of stimulus features/categories
    T = Gamma.shape[0] #number of timepoints
    B_t = np.zeros((n_ch, n_sf, T))
    for it in range(T): #do this for each timepoint
        G_t = Gamma[it,:]
        w_t = np.multiply(B, np.swapaxes(G_t[:,np.newaxis,np.newaxis],0,2) )
        B_t[:,:,it] = np.sum( w_t, axis=-1 )
    
    #Note: B_t.shape is (n_chan, n_stimfeat, n_times)
    Y_pred_ = np.zeros((T, n_sf))
    for it in range(T):
        y_pred = np.matmul(X[it,:], B_t[:,:,it])
        Y_pred_[it,:] = y_pred

    #Note: only Y.shape matters for `continuous_prediction_2class`
    if Y is None: Y=np.zeros((T, Gamma.shape[1]))
    Y_pred = continuous_prediction_2class(Y, Y_pred_)
    
    return Y_pred

#%% Compare ground-truth experimental-defined Y with Ypred
def get_classification_correctness(Y, Ypred):
    '''Given Y and Ypred, return ones and zeros as the time-resolved decoding correctness measure.
    INPUT:
        Y    : array of size (n_timepoints, n_stimfeature)
        Ypred: array of size (n_timepoints, n_stimfeature)
    
    OUTPUT:
        correct: array of size (n_timepoints, 1)
    '''

    # predicted stimulus feature
    Yindex_pred = [ [list(Ypred[it,:]).index(1)] for it in range(Ypred.shape[0]) ]
    Ylabel_pred = np.hstack(Yindex_pred) + 1 #because iTrailImg=1 was coded as 0 when `row_to_matrix`
    
    # ground-truth stimulus feature in the experiment
    Yindex = [ [list(Y[it,:]).index(1)] for it in range(Y.shape[0]) ]
    Ylabel = np.hstack(Yindex) + 1 #because iTrailImg=1 was coded as 0 when `row_to_matrix`

    #compare
    correct = (Ylabel==Ylabel_pred).astype(int)

    return correct[:, np.newaxis]


#%% From row vector to stim-feature matrix
def row_to_matrix(vector):
    # Get unique values from the vector
    unique_values = sorted(set(vector))
    
    # Create an empty array with rows for each value in the vector and columns for each unique value
    design_matrix = np.zeros((len(vector), len(unique_values)))
    
    # Iterate over each unique value
    for idx, value in enumerate(unique_values):
        # Set the corresponding column to 1 where the value matches
        design_matrix[:, idx] = np.array(vector) == value
    
    return design_matrix


#%% Copy-pasted from glhmm.glhmm.preproc, with edits
def apply_pca(X,d,whitening=False,exact=True,pcamodel=None):
    """Applies PCA to the input data X.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_parcels)
        The input data to be transformed.
    d : int or float
        If int, the number of components to keep.
        If float, the percentage of explained variance to keep.
        If array-like of shape (n_parcels, n_components), the transformation matrix.
    whitening : bool, default=False
        Whether to whiten the transformed data.
    exact : bool, default=True
        Whether to use full SVD solver for PCA.

    Returns:
    --------
    X_transformed : array-like of shape (n_samples, n_components)
        The transformed data after applying PCA.
    """

    from sklearn.decomposition import PCA

    if type(d) is np.ndarray:
        X -= np.mean(X,axis=0)
        X = X @ d
        if whitening: X /= np.std(X,axis=0)
        return X

    svd_solver = 'full' if exact else 'auto'

    if d >= 1: 
        if pcamodel is None: #train the model first, then apply
            pcamodel = PCA(n_components=d,whiten=whitening,svd_solver=svd_solver)
            pcamodel.fit(X)
        X = pcamodel.transform(X)
    else:
        if pcamodel is None: #train the model first, then apply
            pcamodel = PCA(whiten=whitening,svd_solver=svd_solver)
            pcamodel.fit(X)
        ncomp = np.where(np.cumsum(pcamodel.explained_variance_ratio_)>=d)[0][0] + 1
        X = pcamodel.transform(X)
        X = X[:,0:ncomp]
        d = ncomp

    # sign convention equal to Matlab's
    for j in range(d):
        jj = np.where(np.abs(X[:,j]) == np.abs(np.max(X[:,j])) )[0][0]
        if X[jj,j] < 0: X[:,j] *= -1

    return X, pcamodel



#%% Copy-pasted from glhmm.glhmm.utils.get_state_evoked_response, with edits
def get_state_evoked_response(Gamma, indices, compute_evoked=True, condition=None):
    """Calculates the state evoked response 

    The first argument can also be a viterbi path (vpath).

    Parameters:
    ---------------
    Gamma : array-like of shape (n_samples, n_states), or a vpath array of shape (n_samples,)
        The Gamma represents the state probability timeseries and the vpath represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    ------------
    ser : array-like of shape (n_samples, n_states)
        The state evoked response matrix.

    Raises:
    -------
    Exception
        If the input data violates any of the following conditions:
        - There is only one trial/session
        - Not all trials/sessions have the same length.
    """

    N = indices.shape[0]
    if N == 1: 
        raise Exception("There is only one segment / trial")
    T = indices[:,1] - indices[:,0]
    if not(np.all(T[0]==T)):
        raise Exception("All segments / trials must have the same length")
    K = Gamma.shape[1]
    T = T[0]

    resp = np.reshape(Gamma,(T,N,K),order='F') #resp.shape is (n_samples, n_epochs, n_states)
    if compute_evoked:
        if condition is None:
            ser = dict(grandavg=np.mean(resp,axis=1))
        else:
            conds = np.unique(condition)
            ser = dict()
            for c in conds:
                cond_sel = condition==c
                cond_avg = np.mean(resp[:,cond_sel,:], axis=1)
                ser[c] = cond_avg
        return ser
    else:
        return resp



#%% Compute condition-specific FO
def get_conditioned_metric(vpath, indices, condition, metric="FO"):
    '''Computes condition-specific state FO, state life times, or state onsets.
    INPUT:
        vpath    : viterbi path returned by `hmm.decode()`
        indices  : array-like of shape (n_sessions/n_trials, 2)
        condition: condition vector array of (n_trials, )
        metric   : "FO", "life_times", or "state_onsets"
    
    OUTPUT:
        out: a dictionary with keys corresponding to condition names
                            values corresponding to condition-averaged metric
    '''

    conds = np.unique(condition)
    out = dict()
    for c in conds:
        #select the trials
        cond_sel = condition==c

        #compute the metric of interest
        if metric=="FO":
            fo_ = utils.get_FO(vpath, indices=indices[cond_sel,:]) #fo_.shape is (n_trials, n_states)
            out[c] = np.mean(fo_, axis=0)
        elif metric=="life_times":
            meanLF, medianLF, maxLF = utils.get_life_times(vpath, indices=indices[cond_sel,:]) #mLF.shape is (n_trials, n_states)
            out[c] = np.mean(medianLF, axis=0)
        elif metric=="state_onsets":
            onsets = utils.get_state_onsets(vpath, indices=indices[cond_sel,:]) #onsets is a list of n_trials
            N = len(onsets) #N trials
            K = len(onsets[0]) #onsets[0] is a numpy array of size n_states
            out[c] = np.ones((K,)) * np.nan
            for ik in range(K):
                on_samp = [ onsets[i][ik] for i in range(N) if onsets[i][ik]!=[] ]
                out[c][ik] = np.median(on_samp)
    return out
