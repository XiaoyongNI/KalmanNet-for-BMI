## ---- LIBRARY OF HELPER FUNCTIONS FOR SIMPLE KALMAN NETS ---- ##
## Necessary libraries for KalmanNets with PyTorch
# By Aren Hite
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split
import scipy.stats
from sklearn import preprocessing
import matplotlib.pyplot as plt

## ---- DATALOADING ---- ##
# REQUIRES: .csv data filename, training set proportion (0-1) (default 0.7=70/10/20 split), validation set proportion (0-1) 
# (default 0.1 = 70/10/20 split)
# RETURNS: list of tensors with either input, target data for each group (training, validation, testing)
def importTensorsNoSeq(filename, num_targs=4, train_prop=0.7, valid_prop=0.1, normalize=True, norm_method="L2", chan_select=False, good_chans=[], add_hist=False, hist_size=0):
    all_data = pd.read_csv(filename).to_numpy() # read dataset from csv
    if chan_select:
        all_data = channelTrim(all_data, num_targs, good_chans)
    if normalize:
        all_data = normalizeInputs(all_data, num_targs, norm_method)
    all_data = torch.from_numpy(all_data).float() # convert to tensor
    if add_hist:
        all_data = addNeuralHistory2D(all_data, hist_size, num_targs)
    train, valid, test = dataSplit(all_data, train_prop, valid_prop)
    return train[:,num_targs:], train[:,:num_targs], valid[:,num_targs:], valid[:,:num_targs], test[:,num_targs:], test[:,:num_targs]

# Like importTensorsNoSeq, but separates the data into time sequences of a specified length
# REQUIRES: .csv file of data, desired sequence length, num_targs is the number of prediction parameters, 
# training set proportion (default 0.7), validation set proportion (default 0.1)
# RETURNS: list of tensors with either input, target data for each group (training, validation, testing)
def importSeqTensors(filename, seq_len, num_targs=4, train_prop=0.7, valid_prop=0.1, normalize=False, norm_method="L2", shuffle_training=False, chan_select=False, good_chans=[], add_hist=False, hist_size=0, overlap=0):
    all_data = pd.read_csv(filename).to_numpy() # unnormalized data
    if chan_select: # trim data to include only good input channels
        all_data = channelTrim(all_data, num_targs, good_chans)
    if normalize: # data normalization (if selected)
        all_data = normalizeInputs(all_data, num_targs, norm_method)
    all_data = torch.from_numpy(all_data) # convert to tensor
    num_pts, data_params = all_data.shape

    # determine the number of points that can be used depending on whether or not there's overlap
    step_size = seq_len - overlap
    if overlap != 0: # we have overlap
        num_seq = ((num_pts - seq_len) // step_size) + 1
    else:
        num_seq = num_pts // seq_len
    
    # tensor to store sequenced data
    seq_tens = torch.zeros((num_seq, seq_len, data_params))
    # store sequences in tensor
    idx = 0
    for seq in range(num_seq):
        seq_tens[seq, :, :] = all_data[idx:idx+seq_len, :]
        idx += step_size
    
    # add history features to the data
    if add_hist:
        seq_tens = addNeuralHistory3D(seq_tens, hist_size, num_targs)

    # train/validation/test split
    train, valid, test = dataSplit(seq_tens, train_prop, valid_prop)

    # shuffle training sequences if true
    if shuffle_training:
        train = shuffleTrainingData(train)

    # return list of tensors, in which data and targets are split such that we have
    # [training_data, training_targets, validation_data, validation_targets, test_data, test_targets]
    return train[:,:,num_targs:], train[:,:,:num_targs], valid[:,:,num_targs:], valid[:,:,:num_targs], test[:,:,num_targs:], test[:,:,:num_targs]

# returns only the passed channels
def channelTrim(full_data, input_idx, good_chans):
    good_idxs = [] 
    # fill with target indices
    for i in range(input_idx):
        good_idxs.append(i)
    # subtract 1 to make them indices, then add input_idx to skip target columns
    for i in good_chans:
        good_idxs.append(i+input_idx-1)
    trim_data = full_data[:,good_idxs]
    return trim_data

# REQUIRES: unsequenced data in numpy array, input_idx = index where input columns begin
# RETURNS: numpy array of normalized data
def normalizeInputs(np_data, input_idx, norm_method):
    if norm_method == "L2":
        np_data[:,input_idx:] = preprocessing.normalize(np_data[:,input_idx:]) # normalize input features
    elif norm_method == "Z":
        np_data[:,input_idx:] = preprocessing.normalize(np_data[:,input:], axis=0)
    else:
        raise Exception("unknown normalization method")
    return np_data # return as np array

def shuffleTrainingData(training_tensor):
    num_seq = training_tensor.shape[0]
    shuf_idxs = torch.randperm(num_seq)
    shuf_data = training_tensor[shuf_idxs]
    return shuf_data

# REQUIRES: dataset to be split, training and validation proportions. optional: dimension to split across (default=0)
# RETURNS: tensors with training, validation, and testing data
def dataSplit(data, train_prop, valid_prop, split_dim=0):
    len = data.shape[split_dim]
    train_size = round(len*train_prop)
    data_remainder = len - train_size
    valid_size = round(data_remainder*valid_prop)
    test_size = data_remainder - valid_size
    train, valid, test = data.split([train_size, valid_size, test_size], dim=split_dim)
    return train, valid, test

# function to add history from hist_size timepoints onto the data
# REQUIRES: tensor of all data, either 2D or 3D. if 2D, should have shape [tpts, features]. 
# if 3D, should have shape [num_seq, seq_len, features]. hist_size, the number of timepoints of history to be appended
# RETURNS: matrix with historical features appended 
def addNeuralHistory3D(data, hist_size, num_targs):
    # separate out targets and features
    targs = data[:,:,:num_targs]
    feats = data[:,:,num_targs:]
    padded_feats = torch.nn.functional.pad(feats, (0,0, hist_size, 0))
    num_seq, seq_len, num_feats = feats.shape
    aug_size = 1 + hist_size
    aug_tens = torch.zeros(num_seq, seq_len, aug_size*num_feats)
    # for loops basically handle filling the same idx of all sequences in the same way
    for i in range(seq_len): # row in each sequence
        for h in range(hist_size + 1): # idx we're augmenting data from
            aug_tens[:,i,h*num_feats:(h+1)*num_feats] = padded_feats[:,i+hist_size-h,:]
    return torch.cat([targs,aug_tens], dim=2)

# accomplished the same thing as addNeuralHistory3D(), but with different input dimensions
def addNeuralHistory2D(data, hist_size, num_targs):
    # separate out targets and features
    targs = data[:,:num_targs]
    feats = data[:,num_targs:]
    padded_feats = torch.nn.functional.pad(feats, (0, 0, hist_size, 0))
    num_pts, num_feats = feats.shape
    aug_size = 1 + hist_size
    aug_tens = torch.zeros(num_pts, aug_size*num_feats)
    for i in range(num_pts): # row in each sequence
        for h in range(hist_size + 1): # idx we're augmenting data from
            aug_tens[i,h*num_feats:(h+1)*num_feats] = padded_feats[i+hist_size-h,:]
    return torch.cat([targs,aug_tens], dim=1)

## ---- NOISE MANIPULATION ---- ##
# REQUIRES: 3D tensor of data to add noise to, list of timepoints to add noise to, magnitude to add onto neural signals at each timepoint, 
# num targets (i.e. how many columns to skip)
# NOTES: magn = 0 replaces the neural signal with 0 (rather than adding 0) to simulate disconnection, timepoints must be < len(data)
# RETURNS: tensor of noisy data
def addNoise(data, tpts, mag):
    noisy_data = data.clone()
    seq_len = data.shape[1] # second dimension
    for t in tpts:
        # determine index
        batch_num = t // seq_len
        idx = t % seq_len
        if mag == 0:
            noisy_data[batch_num, idx, :] = 0
        else:
            noisy_data[batch_num, idx, :] += mag
    return noisy_data

## ---- DATA ANALYSIS ---- ##
# mse for each parameter
# REQUIRES: truth and pred are lists of tensors, where each list element represents a parameter. Each array should be 1 dim
# RETURNS: numpy array of MSE values, where each column is MSE for each param
def mse_all(targ, pred, numpy=False):
    mse = []
    mse_fn = nn.MSELoss()
    for i in range(len(targ)):
        cur_mse = mse_fn(targ[i], pred[i])
        if numpy: # return as numpy arrays
            cur_mse = cur_mse.detach().numpy()
        mse.append(float(cur_mse))
    return mse
  
# correlation coefficient for each parameter
# REQUIRES: truth and pred are lists of tensors, where each list element represents a parameter. Each array should be 1 dim
# RETURNS: numpy array of MSE values, where each column is MSE for each param
def corr_all(targ, pred):
    corrs = []
    for i in range(len(targ)):
        nptarg = targ[i].squeeze().detach().numpy()
        nppred = pred[i].squeeze().detach().numpy()
        corr, p_junk = scipy.stats.pearsonr(nptarg, nppred)
        corrs.append(corr)
    return corrs

# ground truth vs. prediction plot
# REQUIRES: truth and pred are lists of tensors, where each list element represents a parameter. Each array should be 1 dim, 
# lists should be equal length. labels is a list of parameter labels for each graph. pts_graphed is the number of points to include
# on the graph (default = 100)
# RETURNS: numpy array of MSE values, where each column is MSE for each param
def truthPlot(targ, preds, labels, pts_graphed=100, title="Ground Truth vs. Prediction", add_vlines=False, vlines=[]):
    num_params = len(targ)
    fig, axs = plt.subplots(num_params)
    for i in range(num_params):
        plt_targ = targ[i].detach().numpy()
        plt_preds = preds[i].detach().numpy()
        axs[i].plot(plt_targ[0:pts_graphed], label='Ground truth')
        axs[i].plot(plt_preds[0:pts_graphed], label='Prediction')
        if add_vlines:
            for x in vlines:
                axs[i].axvline(x, color="r")
        axs[i].set_title(f'{labels[i]}')
        axs[i].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# barplot for each parameter
# REQUIRES: lists of data and corresponding labels to be plotted, title
# RETURNS: barplot, formatted to look nice :)
def barPlot(data, labels, title="Default Barplot", size=4):
    # convert data to strings in copied list to use as bar labels
    str_data = []
    for i in data:
        str_data.append(str(round(i,2)))
    plt.figure()
    plt.grid(which="major", axis="both", zorder=0)
    colors = ['red', 'orange', 'green', 'lightblue']
    if size == 2:
        colors = ['red', 'lightblue']
    bar = plt.bar(labels, data, color=colors, zorder=3)
    plt.bar_label(bar, str_data)
    plt.title(title)
    plt.show()

